
# Fluorescence BPM Simulation

!(assets/fluo.png)


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
