# cells_dense_noisefree

Ground-truth dataset, main variant, NOISE OFF. 2000 spherical cells (radius 5-9 um, so 10-18 um across) with lognormal brightness in a 256 x 256 x 256 um volume, imaged at NA 0.5 and delivered on a 1 x 1 x 4 um voxel grid with shot and read noise. Cells refract (dn = 0.02) so the PSF degrades with depth; the medium is index-matched.

## As run

- git rev `f6387c0`, seed `42`
- 2000 objects, fill fraction 15.87%
- delivered stack `[64, 256, 256]` (z, y, x) at voxel 1.0 x 1.0 x 4.0 um
- lateral resolution 0.53 um, axial 5.45 um
- peak 305 photons; no detector noise
- ran in 16 s

## Measured against ground truth

- detection efficiency, surface / deepest octant: 0.77
- emission recovery correlation: 0.672
- brightest voxel vs true centre: lateral median 2.24 um, axial median 4.0 um (bounded below by the 4 um voxel)
- nearest-neighbour distance median 15.3 um, 0.0% closer than the axial resolution

## Files

- `fluo_with_noise.gif`, `fluo_without_noise.gif` — green z-sweeps
- `fluo_projections.png` — xy and xz projections of truth, stack, noisy stack
- `analysis.md` — detection efficiency, emission recovery, offsets
- `gt_cells.csv` — per-cell ground truth (position, radius, brightness)
- `config.yaml` — the configuration as run; `code/` — the code that ran it

Volumes (`fluo_with_noise.tif`, `fluo_without_noise.tif`, `fluo_gt.tif`, `fluo_labels.tif`) stay in `log/cells_dense_noisefree/`.

Reproduce with:

```bash
python simulate.py -c config/cells_dense_noisefree.yaml
```
