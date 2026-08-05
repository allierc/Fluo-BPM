# Running the simulator

A short guide to generating stacks with this simulator: the three sweeps that are set
up, and how to change what the sample and the microscope are.

The physics and the measurements behind the defaults are in
[archive/note/note.pdf](archive/note/note.pdf) (one page of model, then three sweeps).
The reference acquisition the defaults follow is in [ACQUISITION.md](ACQUISITION.md).

## Setup

```bash
conda env create -f environment.yaml     # or use an env with torch, scikit-image,
conda activate fluorescence-bpm          # tifffile, pydantic, pyyaml, matplotlib
```

Everything below runs on one GPU. A sweep arm takes 30 s to 2 min at the sizes shipped.

## The three sweeps

Each is one spec file. Arms land in `log/<sweep name>/<arm label>/`.

```bash
python run_sweep.py -s config/sweeps/MC_sweep.yaml     # Monte-Carlo draws: 30 ... 1920
python run_sweep.py -s config/sweeps/RI_sweep.yaml     # cell index: 0 ... 0.02
python run_sweep.py -s config/sweeps/PSF_sweep.yaml    # NA: 1.0 ... 0.3

python run_sweep.py -s config/sweeps/RI_sweep.yaml --dry_run   # list arms, run nothing
python run_sweep.py -s config/sweeps/RI_sweep.yaml --device cuda:0
```

**MC_sweep** — how many random phase draws `W` the incoherent average needs. Incoherent
emission is simulated by averaging coherent propagations, so a finite `W` leaves
residue that looks exactly like detector noise: 9.5% of the local mean at `W = 30`
with the detector switched off, falling as `1/sqrt(W)` to 2.0% at `W = 1920`. Use
`W = 240` for a working number, `W = 1920` when you need clean images. **Grain in a run
with no measurement noise is this, not the camera.**

**RI_sweep** — what the cells' refractive index does. The objective sits past the exit
face, so light from the far plane crosses the whole sample and arrives degraded while
the near plane does not. Across `dn = 0 ... 0.02` at 384 µm depth, cell/gap contrast
near the objective holds (0.060 → 0.045) while the far side collapses (0.046 → 0.000).
Note that *brightness* barely moves: a `dn = 0.02` sphere deflects light ~2°, well
inside NA 1.0's 48.7° collection cone, so light is aberrated rather than lost.

**PSF_sweep** — how the PSF stretches. Lateral FWHM goes as `0.5 λ/NA` but axial as
`λ/(n − sqrt(n² − NA²))`, so lowering NA elongates z much faster than xy: apparent cell
extent stays 8 µm laterally and grows 14 → 36.5 µm axially over NA 1.0 → 0.3.

## Running a single configuration

```bash
python simulate.py -c config/base/reference_optics.yaml -o log/my_run
python simulate.py -c config/base/reference_optics.yaml --n_iterations 240 --seed 7
```

## What lands in a run folder

| file | contents |
|---|---|
| `fluo.tif` | the delivered stack, uint16 ADU. One stack per folder — whether measurement noise is on is a property of the configuration, so the folder name carries it, not the filename |
| `fluo_gt.tif` | true fluorophore concentration on the delivered voxel grid |
| `fluo_labels.tif` | per-cell label map, for attributing a recovered blob to a cell |
| `gt_cells.csv` | one row per cell: `label, x_um, y_um, z_um, r_um, brightness, x_vox, y_vox, z_vox` |
| `fluo.gif` | green z-sweep |
| `fluo_projections.png` | xy and xz projections beside the ground truth — the xz panel is where a top/bottom difference shows |
| `config.yaml` | the configuration as run, including the seed |
| `summary.md` / `.json` | resolution, photon statistics, Monte-Carlo speckle, timing |

Conventions (there are regression tests for these): arrays are `[z, y, x]`, image plane
`k` is focused on sample plane `k`, and voxel `(k, j, i)` is centred at
`x = (i+0.5)·dx`, `y = (j+0.5)·dx`, `z = (k+1)·dz`.

## Changing the sample

In a config's `phantom:` block, or as a sweep parameter (`phantom.<field>`):

| field | meaning |
|---|---|
| `type` | `cells` (scattered), `spheroid` (packed in one ball), `beads` (point emitters on a grid, for probing the PSF) |
| `n_cells`, `radius_um` | how many, how big. Rejection sampling stops at the packing limit and says so — asking for more than the geometry holds is reported, not silently truncated |
| `min_gap_um` | surface-to-surface spacing; 0.1 is touching |
| `brightness` | `lognormal` / `uniform` / `constant`, with `sigma` and `clip` |
| `dn_cell` | mean index contrast of a cell against the medium |
| `dn_noise_rms`, `dn_noise_um` | index texture *inside* the cells, at that correlation length. A smooth homogeneous sphere is a weak low-order lens: swept alone from 0 to 0.10 at NA 1.0 it changed the depth asymmetry by nothing at all (1.00 → 1.01). Sub-cellular structure is what scatters |
| `hollow`, `shell_um` | fluorescence in a shell rather than a filled body |

`medium:` puts index structure outside the cells: `smooth_random` (a band-limited
landscape, `dn_rms` and `correlation_um`) or `slab` (a tilted mismatched layer).

## Changing the imaging

| block | field | meaning |
|---|---|---|
| `optics` | `na`, `lbda`, `nm` | numerical aperture, emission wavelength, immersion index |
| `optics` | `zernike` | pupil aberration in waves RMS: `defocus`, `astig_0/45`, `coma_x/y`, `spherical` |
| `volume` | `nx`, `ny`, `nz`, `dx`, `dz` | the **simulation** grid, finer than what is delivered |
| `camera` | `bin_xy`, `bin_z` | binning from the simulation grid to the delivered voxel |
| `camera` | `photons_peak`, `background_photons` | the photon budget, which sets shot noise |
| `camera` | `poisson`, `read_noise_e` | measurement noise on or off |
| `camera` | `gain`, `offset`, `bit_depth` | detector conversion: `gain` is DN per photo-electron, so `2^bit_depth / gain` is where it saturates |
| `emission` | `n_iterations` | the `W` above |
| `emission` | `axial_incoherent` | leave `true`. `false` shares one emission phase screen down the whole volume, which makes each cell's slices mutually coherent and rings every cell centre with Fresnel fringes that averaging cannot remove |

**One constraint the engine enforces:** `dx < lbda / (2·NA)`. Sampled coarser than
that, the pupil mask is wider than the represented k-space, nothing is filtered, and
the model has no optical sectioning at all — every plane comes out as the same
projection. The error message reports the maximum usable NA for your `dx`. This is why
the simulation grid is finer than the delivered pixel: the reference's 259 nm pixel is
itself exactly at the diffraction cutoff.

## Measuring a run

```bash
python analyze_run.py log/RI_sweep/dn0100      # -> analysis.md, analysis.json
```

Reports detection efficiency against depth, brightness recovery, localization offset,
crowding, and the near/far sharpness ratio. Two cautions learned from using it:

- The **gradient-energy** sharpness ratio is dominated by fine haze texture in a dense
  stack and will report 1.00–1.01 even where cell contrast has collapsed to zero. For
  depth degradation, trust cell/gap contrast (see the note).
- Any width measured on these stacks must be taken at half maximum **above the local
  floor**. Measured above zero, the widefield haze pedestal inflates an 8 µm cell to
  12 µm laterally.

## Tests

```bash
python test_registration.py      # image plane k is focused on sample plane k
python test_determinism.py       # same seed -> byte-identical output
python test_depth_sharpness.py   # near/far asymmetry, with dn = 0 as the control
```

## Rebuilding the note

```bash
python note.py                   # -> archive/note/note.pdf, from the three log/ sweeps
```
