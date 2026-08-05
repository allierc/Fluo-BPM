# cells_real_like

Matched by eye to a real widefield plane the user supplied: cells spanning about an eighth of the frame, nearly touching, a wide brightness spread with a few near-saturated cells, heavy out-of-focus haze, and visible grain on that haze. Scaled from the reference by pixel count rather than by microns: cells are ~30 px in diameter with ~10 across the frame, which on the 1 um delivered grid of the other configurations means a 320 px field and 24-32 um cells.

## As run

- git rev `0d42420`, seed `11`
- 130 objects, fill fraction 21.10%
- delivered stack `[16, 320, 320]` (z, y, x) at voxel 1.0 x 1.0 x 4.0 um
- lateral resolution 0.53 um, axial 5.45 um
- peak 306 photons; bright-voxel SNR 7.8
- ran in 6 s

## Measured against ground truth

- detection efficiency, surface / deepest octant: 1.07
- emission recovery correlation: 0.872
- brightest voxel vs true centre: lateral median 2.00 um, axial median 12.0 um (bounded below by the 4 um voxel)
- nearest-neighbour distance median 28.4 um, 0.0% closer than the axial resolution
- noise check: measured std 6.1 vs shot+read prediction 6.1 (ratio 1.00)

## Files

- `fluo_with_noise.gif`, `fluo_without_noise.gif` — green z-sweeps
- `fluo_projections.png` — xy and xz projections of truth, stack, noisy stack
- `analysis.md` — detection efficiency, emission recovery, offsets
- `gt_cells.csv` — per-cell ground truth (position, radius, brightness)
- `config.yaml` — the configuration as run; `code/` — the code that ran it

Volumes (`fluo_with_noise.tif`, `fluo_without_noise.tif`, `fluo_gt.tif`, `fluo_labels.tif`) stay in `log/cells_real_like/`.

Reproduce with:

```bash
python simulate.py -c config/cells_real_like.yaml
```
