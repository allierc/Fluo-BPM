# cells_spheroid

Spheroid rather than dispersed cells: 550 larger cells (radius 5-7 um) packed inside one 70 um ball at the centre of the volume. Light reaching the far side has crossed the whole aggregate, so this is the variant where depth-dependent scattering dominates.

## As run

- git rev `587705f`, seed `7`
- 550 objects, fill fraction 2.54%
- delivered stack `[64, 256, 256]` (z, y, x) at voxel 1.0 x 1.0 x 4.0 um
- lateral resolution 0.53 um, axial 5.45 um
- peak 305 photons; bright-voxel SNR 12.9
- ran in 18 s

## Measured against ground truth

- detection efficiency, surface / deepest octant: 0.86
- emission recovery correlation: 0.529
- brightest voxel vs true centre: lateral median 2.83 um, axial median 4.0 um (bounded below by the 4 um voxel)
- nearest-neighbour distance median 11.8 um, 0.0% closer than the axial resolution
- noise check: measured std 7.9 vs shot+read prediction 7.8 (ratio 1.00)

## Files

- `fluo_with_noise.gif`, `fluo_without_noise.gif` — green z-sweeps
- `fluo_projections.png` — xy and xz projections of truth, stack, noisy stack
- `analysis.md` — detection efficiency, emission recovery, offsets
- `gt_cells.csv` — per-cell ground truth (position, radius, brightness)
- `config.yaml` — the configuration as run; `code/` — the code that ran it

Volumes (`fluo_with_noise.tif`, `fluo_without_noise.tif`, `fluo_gt.tif`, `fluo_labels.tif`) stay in `log/cells_spheroid/`.

Reproduce with:

```bash
python simulate.py -c config/cells_spheroid.yaml
```
