
# Fluorescence BPM Simulation

![ouput](assets/fluo.gif) 


Differentiable beam propagation method (BPM) for fluorescence microscopy simulation in heterogeneous biological tissue.

## Physics

**Split-step propagation:**
```
E(z+dz) = F⁻¹[F[E(z)·exp(ik₀·Δn·dz)]·H(dz)]
```

**Fresnel propagator:**
```
H(k_x,k_y) = exp(2πi·dz·√((n/λ)² - k_x² - k_y²))
```

**Incoherent fluorescence (Monte Carlo):**
```
I_total = Σ |F⁻¹[P(k)·F[√(F(r))·exp(iφ_random)]]|²
```

## Installation

```bash
conda env create -f environment.yaml
conda activate fluorescence-bpm
pip install -e .
```

## Usage

**Quick start (default parameters):**
```python
from fluorescence_bpm import run_simulation

run_simulation(
    refractive_index_path='./data/KidneyL_30.tif',
    fluorescence_path='./data/Fluo_2_30.tif',
    n_iterations=1000
)
```

**Custom parameters:**
```python
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
```

**Using Config object:**
```python
from fluorescence_bpm import FluorescenceBPM, Config
import torch

config = Config(nm=1.33, na=0.5, dx=0.1154)
model = FluorescenceBPM(config)

phi = torch.rand((512, 512), device='cuda:0') * 2 * torch.pi
I = model(field_number=0, phi=phi)
```

## Dataset engine (config driven)

For generating ground-truth datasets — a volume filled with spherical cells of
differing brightness, imaged and delivered on a chosen voxel grid — one YAML
describes the whole experiment and the engine writes one log folder:

```bash
python simulate.py -c config/cells_dense.yaml     # -> log/cells_dense/
./run_all.sh                                      # every config in config/
python analyze_run.py log/*                        # measure against ground truth
python archive_runs.py                             # freeze into archive/
```

Parameterized sweeps are one spec file each:

```bash
python run_sweep.py -s config/sweeps/MC_sweep.yaml     # Monte-Carlo draws
python run_sweep.py -s config/sweeps/RI_sweep.yaml     # cell refractive index
python run_sweep.py -s config/sweeps/PSF_sweep.yaml    # numerical aperture
```

The YAML covers the volume and phantom (cells, spheroid or a bead grid), the
refractive-index landscape of the medium and inside the cells, the optics (NA,
wavelength, Zernike pupil aberration), the emission Monte Carlo, and the detector
(binning to the delivered voxel, Poisson and read noise).

- [RUNNING.md](RUNNING.md) — how to run the sweeps and change the sample or the imaging
- [archive/note/note.pdf](archive/note/note.pdf) — the forward model in one page, then
  the three sweeps measured against it
- [ACQUISITION.md](ACQUISITION.md) — the reference acquisition the defaults follow
- [DATASET.md](DATASET.md) — dataset conventions and measured properties

Two constraints the engine enforces or documents, both learned the hard way:

- `volume.dx` must satisfy `dx < lbda/(2*NA)`, else the pupil mask is larger than
  the sampled k-space, passes everything, and the simulation has no optical
  sectioning — the configuration is rejected with the maximum usable NA.
- Image plane `k` is focused on sample plane `k`. The core returns the focal stack
  indexed from the exit face inward, so an unflipped stack is mirrored in z
  against the sample; `test_registration.py` pins this.

Runs are byte-for-byte repeatable at a fixed `seed` (`test_determinism.py`).

## Data Format

Place TIFF files in `./data/`:
- **KidneyL_30.tif**: Refractive index volume (Δn values)
- **Fluo_2_30.tif**: Fluorophore concentration volume

TIFF format: `[z, y, x]` → code transposes to `[x, y, z]`

## Output

- `fluorescence_final.tif`: Final accumulated intensity
- `fluo_iter_XXXX.tif`: Intermediate checkpoints (every 100 iterations)

## Theory

### Propagation equation

Helmholtz equation with varying n(r):
```
∇²E + k₀²n²(r)E = 0
```

BPM approximation (paraxial):
```
∂E/∂z = i/(2k₀n₀)[∇_⊥²E + k₀²(n²-n₀²)E]
```

### Split-step method

1. **Phase accumulation**: `E' = E·exp(ik₀·Δn·dz)`
2. **Diffraction**: `E(z+dz) = F⁻¹[F[E']·H(dz)]`
3. **Pupil filtering**: `E_det = F⁻¹[P·F[E]]` where P = pupil function

### Fluorescence modeling

Incoherent sum over random phase realizations:
```
I(r) = ⟨|E_emission(r,φ)|²⟩_φ
```

## Citation

Based on optical diffraction tomography and beam propagation methods for thick biological samples.

## License

Janelia Open-Source License
