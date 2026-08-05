"""Configuration schema for the Fluo-BPM dataset engine.

One YAML describes a whole experiment: the volume and what is in it (phantom),
the medium it sits in, the optics, the emission Monte Carlo, and the detector.
The engine reads exactly this and writes one log folder.

    from fluo_bpm.config import FluoBPMConfig
    config = FluoBPMConfig.from_yaml('config/cells_dense.yaml')
"""
from enum import Enum
from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class StrEnum(str, Enum):
    pass


class PhantomType(StrEnum):
    CELLS = "cells"            # spheres scattered through the whole volume
    SPHEROID = "spheroid"      # spheres packed inside one large ball
    BEADS = "beads"            # sub-resolution point emitters on a grid (PSF probe)


class MediumType(StrEnum):
    NONE = "none"              # index-matched: shift-invariant PSF
    SMOOTH_RANDOM = "smooth_random"   # band-limited random index landscape
    SLAB = "slab"              # tilted index slab across the top of the volume


class BrightnessDist(StrEnum):
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    CONSTANT = "constant"


class VolumeConfig(BaseModel):
    """The *simulation* grid. Keep dx fine enough that the pupil fits in k-space:
    the model can only represent NA <= lbda / (2*dx). The detector grid is coarser
    (see CameraConfig.bin_xy / bin_z)."""
    model_config = ConfigDict(extra="forbid")

    nx: int = 512
    ny: int = 512
    nz: int = 256
    dx: float = 0.5            # lateral sampling of the simulation [um]
    dz: float = 1.0            # axial step of the split-step propagation [um]

    @property
    def extent_um(self) -> Tuple[float, float, float]:
        return (self.nx * self.dx, self.ny * self.dx, self.nz * self.dz)


class BrightnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dist: BrightnessDist = BrightnessDist.LOGNORMAL
    sigma: float = 0.6         # lognormal sigma (log units)
    low: float = 0.3           # uniform range / constant value
    high: float = 3.0
    clip: Tuple[float, float] = (0.08, 4.0)


class PhantomConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: PhantomType = PhantomType.CELLS
    n_cells: int = 3000
    radius_um: Tuple[float, float] = (3.0, 6.0)
    min_gap_um: float = 0.5            # surface-to-surface spacing
    edge_um: float = 0.4               # membrane softness of the tanh edge
    dn_cell: float = 0.02              # index contrast of a cell vs medium
    dn_noise_rms: float = 0.0          # index texture INSIDE the cells (organelles).
                                       # A homogeneous sphere only refracts, weakly and
                                       # at low order; real scattering comes from
                                       # sub-cellular structure, so the random part
                                       # belongs on the cell map, not in the medium.
    dn_noise_um: float = 1.0           # correlation length of that texture
    brightness: BrightnessConfig = BrightnessConfig()
    hollow: bool = False               # fluorescence in a shell only (cytoplasm/membrane)
    shell_um: float = 1.5              # shell thickness when hollow
    margin_um: float = 0.0             # keep cells this far from the volume border
    # spheroid only
    spheroid_radius_um: float = 60.0
    spheroid_center_um: Optional[Tuple[float, float, float]] = None
    # beads only
    bead_grid: Tuple[int, int, int] = (5, 5, 9)   # nx, ny, nz probe positions
    bead_radius_um: float = 0.15
    bead_brightness: float = 1.0

    @model_validator(mode='after')
    def _check(self):
        if self.radius_um[0] > self.radius_um[1]:
            raise ValueError('radius_um must be (min, max)')
        return self


class MediumConfig(BaseModel):
    """Refractive-index landscape the cells are embedded in.

    This is what makes the PSF depend on x and y as well as z: the BPM
    accumulates the phase of whatever it walks through, so a smooth index
    landscape gives a genuinely shift-varying PSF, which a single pupil
    aberration cannot do."""
    model_config = ConfigDict(extra="forbid")

    type: MediumType = MediumType.NONE
    dn_rms: float = 0.004              # rms index deviation of the landscape
    correlation_um: float = 25.0       # lateral correlation length
    correlation_z_um: float = 40.0     # axial correlation length
    slab_dn: float = 0.02              # SLAB: index step
    slab_thickness_um: float = 20.0
    slab_tilt: float = 0.02            # SLAB: fractional thickness ramp across x


class ZernikeConfig(BaseModel):
    """Detection-pupil aberration in waves RMS (0 = diffraction limited).

    Shift-invariant by construction; use MediumConfig for field dependence."""
    model_config = ConfigDict(extra="forbid")

    defocus: float = 0.0
    astig_0: float = 0.0
    astig_45: float = 0.0
    coma_x: float = 0.0
    coma_y: float = 0.0
    spherical: float = 0.0

    def any_nonzero(self) -> bool:
        return any(abs(v) > 0 for v in self.model_dump().values())


class OpticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nm: float = 1.33                   # medium refractive index
    na: float = 0.5                    # detection numerical aperture
    lbda: float = 0.532                # emission wavelength [um]
    directions: List[List[float]] = [[0.0, 0.0, 1.0]]
    zernike: ZernikeConfig = ZernikeConfig()


class EmissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_iterations: int = 200            # Monte Carlo random-phase realizations
    stochastic: bool = False           # STORM/PALM sparse activation
    sparsity: float = 0.01
    axial_incoherent: bool = True      # independent emission phase per plane. False
                                       # reuses one screen for the whole volume, which
                                       # makes each cell's layers mutually coherent and
                                       # rings its centre with Fresnel zones.


class CameraConfig(BaseModel):
    """Detector sampling and noise. bin_xy / bin_z take the simulation grid to the
    voxel size of the delivered dataset."""
    model_config = ConfigDict(extra="forbid")

    bin_xy: int = 2                    # 0.5 um sim -> 1.0 um voxel
    bin_z: int = 4                     # 1.0 um sim -> 4.0 um voxel
    photons_peak: float = 300.0        # scale so the clean volume peaks at this many photons
    background_photons: float = 5.0    # uniform background before noise
    poisson: bool = True
    read_noise_e: float = 2.0          # Gaussian read noise [electrons]
    offset: float = 100.0              # camera offset [ADU]
    gain: float = 1.0                  # ADU per photo-electron
    bit_depth: int = 16


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = 'cuda:0'
    deterministic: bool = False        # pin algorithm selection as well as seeds
    log_dir: str = './log'
    save_fine_volume: bool = False     # the un-binned simulation grid (large)
    save_gt_volume: bool = True        # binned ground-truth fluorophore volume
    save_label_volume: bool = True     # binned per-cell label map
    save_gif: bool = True
    gif_gamma: float = 0.6
    gif_duration_ms: int = 80
    gif_max_frames: int = 120
    psf_report: bool = False           # fit each bead, tabulate PSF vs (x, y, z)


class FluoBPMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = "fluo_bpm"
    seed: Optional[int] = 42           # None = draw from entropy (run not repeatable)
    config_file: str = "none"

    volume: VolumeConfig = VolumeConfig()
    phantom: PhantomConfig = PhantomConfig()
    medium: MediumConfig = MediumConfig()
    optics: OpticsConfig = OpticsConfig()
    emission: EmissionConfig = EmissionConfig()
    camera: CameraConfig = CameraConfig()
    engine: EngineConfig = EngineConfig()

    @model_validator(mode='after')
    def _check_sampling(self):
        nyquist = 1.0 / (2.0 * self.volume.dx)
        cutoff = self.optics.na / self.optics.lbda
        if cutoff >= nyquist:
            raise ValueError(
                f'pupil cutoff NA/lbda = {cutoff:.3f} um^-1 exceeds the Nyquist '
                f'frequency 1/(2*dx) = {nyquist:.3f} um^-1: the pupil mask would pass '
                f'the entire sampled k-space and the simulation would have no optical '
                f'sectioning. Use dx < lbda/(2*NA) = {self.optics.lbda / (2 * self.optics.na):.3f} um.')
        if self.volume.nx % self.camera.bin_xy or self.volume.ny % self.camera.bin_xy:
            raise ValueError('nx, ny must be divisible by camera.bin_xy')
        if self.volume.nz % self.camera.bin_z:
            raise ValueError('nz must be divisible by camera.bin_z')
        return self

    @staticmethod
    def from_yaml(file_name: str) -> 'FluoBPMConfig':
        with open(file_name, 'r') as f:
            raw = yaml.safe_load(f)
        config = FluoBPMConfig(**raw)
        config.config_file = file_name
        return config

    def pretty(self) -> str:
        return yaml.dump(self.model_dump(mode='json'), default_flow_style=False,
                         sort_keys=False, indent=4)


if __name__ == '__main__':
    import sys
    cfg = FluoBPMConfig.from_yaml(sys.argv[1] if len(sys.argv) > 1 else 'config/cells_dense.yaml')
    print(cfg.pretty())
