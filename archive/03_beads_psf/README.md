# beads_psf

PSF probe, ideal case. 5 x 5 x 13 sub-resolution beads on a grid through an index-matched volume with an unaberrated pupil: the PSF should depend on depth only through defocus, and not on x or y. This is the reference the aberrated probe is measured against.

## As run

- git rev `587705f`, seed `3`
- 325 objects, fill fraction 0.00%
- delivered stack `[64, 256, 256]` (z, y, x) at voxel 1.0 x 1.0 x 4.0 um
- lateral resolution 0.53 um, axial 5.45 um
- peak 305 photons; bright-voxel SNR 3.0
- ran in 27 s
- PSF over the bead grid: lateral FWHM 0.53 +- 0.00 um, axial 4.70 +- 0.01 um

## Measured against ground truth

- detection efficiency, surface / deepest octant: 1.00
- emission recovery correlation: n/a (uniform brightness)
- brightest voxel vs true centre: lateral median 0.00 um, axial median 0.0 um (bounded below by the 4 um voxel)
- nearest-neighbour distance median 21.0 um, 0.0% closer than the axial resolution
- noise check: measured std 3.4 vs shot+read prediction 3.4 (ratio 1.00)

## Files

- `fluo_with_noise.gif`, `fluo_without_noise.gif` — green z-sweeps
- `fluo_projections.png` — xy and xz projections of truth, stack, noisy stack
- `analysis.md` — detection efficiency, emission recovery, offsets
- `gt_cells.csv` — per-cell ground truth (position, radius, brightness)
- `config.yaml` — the configuration as run; `code/` — the code that ran it

Volumes (`fluo_with_noise.tif`, `fluo_without_noise.tif`, `fluo_gt.tif`, `fluo_labels.tif`) stay in `log/beads_psf/`.

Reproduce with:

```bash
python simulate.py -c config/beads_psf.yaml
```
