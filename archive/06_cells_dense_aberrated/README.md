# cells_dense_aberrated

Hard variant: same 2000-cell volume, but embedded in a smooth random refractive index landscape (dn_rms 0.004, 25 um correlation) and imaged through an aberrated pupil (0.3 waves spherical, 0.15 astigmatism). The PSF then varies with x, y and z, which is the case a shift-invariant reconstruction cannot fit.

## As run

- git rev `587705f`, seed `42`
- 2000 objects, fill fraction 15.87%
- delivered stack `[64, 256, 256]` (z, y, x) at voxel 1.0 x 1.0 x 4.0 um
- lateral resolution 0.53 um, axial 5.45 um
- peak 305 photons; bright-voxel SNR 12.8
- ran in 27 s

## Measured against ground truth

- detection efficiency, surface / deepest octant: 0.90
- emission recovery correlation: 0.671
- brightest voxel vs true centre: lateral median 3.00 um, axial median 4.0 um (bounded below by the 4 um voxel)
- nearest-neighbour distance median 15.3 um, 0.0% closer than the axial resolution
- noise check: measured std 11.6 vs shot+read prediction 11.6 (ratio 1.00)

## Files

- `fluo_with_noise.gif`, `fluo_without_noise.gif` — green z-sweeps
- `fluo_projections.png` — xy and xz projections of truth, stack, noisy stack
- `analysis.md` — detection efficiency, emission recovery, offsets
- `gt_cells.csv` — per-cell ground truth (position, radius, brightness)
- `config.yaml` — the configuration as run; `code/` — the code that ran it

Volumes (`fluo_with_noise.tif`, `fluo_without_noise.tif`, `fluo_gt.tif`, `fluo_labels.tif`) stay in `log/cells_dense_aberrated/`.

Reproduce with:

```bash
python simulate.py -c config/cells_dense_aberrated.yaml
```
