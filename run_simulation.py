"""
Example: Run fluorescence simulation
"""
from fluorescence_bpm import FluorescenceBPM, Config, run_simulation
import torch



run_simulation(
    refractive_index_path='./data/KidneyL_30.tif',
    fluorescence_path='./data/Fluo_2_30.tif',
    nm=1.33,        # Medium refractive index
    na=0.5,         # Numerical aperture
    dx=0.1154,      # Pixel size [μm]
    lbda=0.5320,    # Wavelength [μm]
    dz=1.0,         # Axial step [μm]
    z_min=-15.0,    # Volume start [μm]
    z_max=15.0,     # Volume end [μm]
    device='cuda:0',
    n_iterations=1000
)


