# beads_psf_aberrated

PSF probe, xyz-varying case. 5 x 5 x 13 sub-resolution beads on a grid through an volume with a smooth random index landscape and an aberrated pupil, so the measured width depends on field position as well as depth. Compare psf_table.csv against the ideal probe.

## As run

- git rev `f6387c0`, seed `3`
- 325 objects, fill fraction 0.00%
- delivered stack `[64, 256, 256]` (z, y, x) at voxel 1.0 x 1.0 x 4.0 um
- lateral resolution 0.53 um, axial 5.45 um
- peak 305 photons; bright-voxel SNR 3.6
- ran in 35 s
- PSF over the bead grid: lateral FWHM 0.71 +- 0.11 um, axial 4.26 +- 0.99 um

## Measured against ground truth

- detection efficiency, surface / deepest octant: 0.50
- emission recovery correlation: n/a (uniform brightness)
- brightest voxel vs true centre: lateral median 1.00 um, axial median 4.0 um (bounded below by the 4 um voxel)
- nearest-neighbour distance median 21.0 um, 0.0% closer than the axial resolution
- noise check: measured std 3.6 vs shot+read prediction 3.5 (ratio 1.00)

## Files

- `fluo_with_noise.gif`, `fluo_without_noise.gif` — green z-sweeps
- `fluo_projections.png` — xy and xz projections of truth, stack, noisy stack
- `analysis.md` — detection efficiency, emission recovery, offsets
- `gt_cells.csv` — per-cell ground truth (position, radius, brightness)
- `config.yaml` — the configuration as run; `code/` — the code that ran it

Volumes (`fluo_with_noise.tif`, `fluo_without_noise.tif`, `fluo_gt.tif`, `fluo_labels.tif`) stay in `log/beads_psf_aberrated/`.

Reproduce with:

```bash
python simulate.py -c config/beads_psf_aberrated.yaml
```
