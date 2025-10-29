"""
Fluorescence microscopy simulation using Beam Propagation Method (BPM)
with heterogeneous light propagation through biological tissue.
"""
import numpy as np
import torch
import torch.nn.functional as nf
from tqdm import tqdm
from skimage import io
from tifffile import imwrite
from dataclasses import dataclass
from typing import List


from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Simulation configuration with default values."""
    nm: float = 1.33              # Medium refractive index
    na: float = 0.5               # Numerical aperture
    dx: float = 0.1154            # Pixel size [μm]
    lbda: float = 0.5320          # Wavelength [μm]
    dz: float = 1.0               # Axial step [μm]
    z_min: float = -15.0          # Volume start [μm]
    z_max: float = 15.0           # Volume end [μm]
    device: str = 'cuda:0'        # Device
    refractive_index_path: str = './data/KidneyL_30.tif'
    fluorescence_path: str = './data/Fluo_2_30.tif'
    directions: List[List[float]] = None  # Illumination directions
    output_path: str = './output'
    
    def __post_init__(self):
        if self.directions is None:
            self.directions = [[0, 0, 1]]  # Default: normal incidence


def sqrt_cpx(t1):
    """Complex square root via real/imaginary decomposition."""
    bs_pos = nf.relu(t1)
    bs_neg = nf.relu(-t1)
    return torch.stack((torch.sqrt(bs_pos), torch.sqrt(bs_neg)), dim=len(t1.shape))


def exp_cpx(input, conj=False):
    """Complex exponential: exp(i*input) or exp(-i*input)."""
    output = input.clone()
    amplitude = torch.exp(input[..., 1] if conj else -input[..., 1])
    output[..., 0] = amplitude * torch.cos(input[..., 0])
    output[..., 1] = amplitude * torch.sin(input[..., 0])
    return output


class FluorescenceBPM(torch.nn.Module):
    """
    Beam Propagation Method for fluorescence microscopy simulation.
    
    Models:
    - Heterogeneous light propagation (varying refractive index)
    - Incoherent fluorescence emission
    - Pupil function filtering
    - Split-step Fourier propagation
    """
    
    def __init__(self, config: Config = None):
        super().__init__()
        
        if config is None:
            config = Config()
        self.config = config
        
        # Optical parameters
        self.nm = config.nm
        self.na = config.na
        self.dx = config.dx
        self.lbda = config.lbda
        self.dz = config.dz
        self.z_min = config.z_min
        self.z_max = config.z_max
        
        # Device
        self.device = config.device
        self.dtype = torch.float32
        
        # Load data
        dn_volume = io.imread(config.refractive_index_path)
        dn_volume = np.moveaxis(dn_volume, 0, -1)
        self.dn = torch.tensor(dn_volume, device=self.device, dtype=self.dtype, requires_grad=False)
        
        fluo_volume = io.imread(config.fluorescence_path)
        fluo_volume = np.moveaxis(fluo_volume, 0, -1)
        self.fluo = torch.tensor(fluo_volume, device=self.device, dtype=self.dtype, requires_grad=False)
        
        # Illumination directions
        self.directions = config.directions
        
        # Volume parameters
        self.z_volume = np.arange(self.z_min, self.z_max, self.dz)
        self.Nz = len(self.z_volume)
        self.Nx, self.Ny = dn_volume.shape[:2]
        
        # Spatial frequencies
        int_x = np.arange(-self.Nx/2, self.Nx/2)
        int_y = np.arange(-self.Ny/2, self.Ny/2)
        x_range = self.Nx * self.dx
        y_range = self.Ny * self.dx
        mux = np.fft.fftshift(int_x / x_range).reshape(1, -1)
        muy = np.fft.fftshift(int_y / y_range).reshape(-1, 1)
        
        self.mux = torch.tensor(mux, dtype=self.dtype, device=self.device)
        self.muy = torch.tensor(muy, dtype=self.dtype, device=self.device)
        
    def fresnel_propagator(self, dz, direction=[0,0,1]):
        """
        Compute Fresnel propagator in k-space.
        
        H(k_x, k_y) = exp(i * 2π * dz * sqrt((n/λ)² - k_x² - k_y²))
        """
        K = (self.nm / self.lbda)**2 - (self.mux - direction[0])**2 - (self.muy - direction[1])**2
        hz = exp_cpx(2 * np.pi * dz * sqrt_cpx(K), conj=(dz <= 0))
        return hz
        
    def forward(self, field_number=0, phi=None):
        """
        Forward propagation with fluorescence emission.
        
        Args:
            field_number: Index of illumination direction
            phi: Random phase [Nx, Ny] for incoherent emission
            
        Returns:
            I: Intensity volume [Nx, Ny, Nz]
        """
        # Illumination direction
        fdir = self.directions[field_number]
        k0 = 2 * np.pi / self.lbda
        
        # Compute pupil function
        mux_inc = self.mux - fdir[0]
        muy_inc = self.muy - fdir[1]
        munu = torch.sqrt(mux_inc**2 + muy_inc**2).reshape(self.Nx, self.Ny, 1)
        pupil_mask = (munu < (self.na / self.lbda)).float().squeeze()
        
        pupil = torch.complex(
            pupil_mask, 
            torch.zeros_like(pupil_mask)
        )
        
        # Angle correction
        muxy = np.sqrt(fdir[0]**2 + fdir[1]**2) * self.lbda / self.nm
        theta = np.arcsin(muxy)
        cos_theta = np.cos(theta)
        
        # Propagators
        Hz_forward = torch.view_as_complex(self.fresnel_propagator(self.dz, fdir).detach())
        Hz_backward = torch.view_as_complex(self.fresnel_propagator(-self.dz, fdir).detach())
        
        # Initialize field
        field = torch.zeros((self.Nx, self.Ny), dtype=torch.cfloat, device=self.device)
        coef = torch.tensor(self.dz * k0 / cos_theta * 1.j, dtype=torch.cfloat, device=self.device)
        
        dn_layers = self.dn.unbind(dim=2)
        fluo_layers = self.fluo.unbind(dim=2)
        
        # Forward propagation through volume
        for i in range(self.Nz):
            # Phase from refractive index
            depha = field * torch.exp(dn_layers[i] * coef)
            
            # Fluorescence source term
            S = torch.sqrt(fluo_layers[i]) * torch.exp(phi * 1.j)
            S = torch.fft.ifftn(torch.fft.fftn(S) * pupil_mask)
            
            # Split-step: source + propagation
            field = torch.fft.ifftn(torch.fft.fftn(depha + S) * Hz_forward)
        
        # Apply pupil in k-space
        field = torch.fft.fftn(field) * pupil
        
        # Backward propagation to detector
        I = torch.zeros_like(self.dn)
        for i in range(self.Nz):
            I[:,:,i] = torch.abs(torch.fft.ifftn(field))**2
            field = field * Hz_backward
            
        return I


def run_simulation(
    refractive_index_path: str = './data/KidneyL_30.tif',
    fluorescence_path: str = './data/Fluo_2_30.tif',
    output_path: str = './output',
    n_iterations: int = 1000,
    nm: float = 1.33,
    na: float = 0.5,
    dx: float = 0.1154,
    lbda: float = 0.5320,
    dz: float = 1.0,
    z_min: float = -15.0,
    z_max: float = 15.0,
    device: str = 'cuda:0',
    directions: List[List[float]] = None
):
    """
    Run fluorescence simulation with Monte Carlo sampling.
    
    Args:
        refractive_index_path: Path to refractive index TIFF
        fluorescence_path: Path to fluorophore concentration TIFF
        output_path: Directory for output files
        n_iterations: Number of random phase realizations
        nm: Medium refractive index
        na: Numerical aperture
        dx: Pixel size [μm]
        lbda: Wavelength [μm]
        dz: Axial step [μm]
        z_min: Volume start [μm]
        z_max: Volume end [μm]
        device: PyTorch device
        directions: List of illumination directions [[x,y,z], ...]
    """
    config = Config(
        nm=nm, na=na, dx=dx, lbda=lbda, dz=dz,
        z_min=z_min, z_max=z_max, device=device,
        refractive_index_path=refractive_index_path,
        fluorescence_path=fluorescence_path,
        directions=directions,
        output_path=output_path
    )
    
    # Initialize model
    model = FluorescenceBPM(config)
    
    # Accumulate incoherent intensity
    I_total = torch.zeros_like(model.dn)
    
    print(f"Running {n_iterations} iterations...")
    for n in tqdm(range(n_iterations)):
        # Random phase for incoherent emission
        phi = torch.rand(
            (model.Nx, model.Ny), 
            dtype=torch.float32, 
            device=device
        ) * 2 * np.pi
        
        with torch.no_grad():
            I_total += model(field_number=0, phi=phi)
            
        # Save intermediate results
        if n % 100 == 0 and n > 0:
            I_cpu = I_total.cpu().numpy()
            imwrite(f"{output_path}/fluo_iter_{n:04d}.tif", 
                   np.moveaxis(I_cpu, -1, 0))
    
    # Save final result
    I_final = I_total.cpu().numpy()
    imwrite(f"{output_path}/fluorescence_final.tif", 
           np.moveaxis(I_final, -1, 0))
    
    print("Simulation complete!")
    return I_final


if __name__ == '__main__':
    run_simulation(n_iterations=1000)
