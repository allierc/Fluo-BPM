# beads_psf_aberrated

PSF probe, xyz-varying case. 5 x 5 x 13 sub-resolution beads on a grid through an volume with a smooth random index landscape and an aberrated pupil, so the measured width depends on field position as well as depth. Compare psf_table.csv against the ideal probe.


- git rev `f6387c0`, seed `3`
- simulation grid 512 x 512 x 256 at 0.5 / 1.0 um
- delivered stack [64, 256, 256] at voxel 1.0 x 1.0 x 4.0 um
- 325 objects, fill fraction 0.00%
- NA 0.5, lambda 0.532 um, n 1.33; lateral resolution 0.53 um, axial 5.45 um
- 400 Monte Carlo realizations
- medium `MediumType.SMOOTH_RANDOM`, cell dn 0.0
- Poisson noise on, read noise 2.0 e, peak 305 photons, bright-voxel SNR 3.6
- timing: phantom 10.1 s, simulation 20.3 s, total 35.1 s
- PSF over the bead grid: lateral FWHM 0.71 +- 0.11 um, axial 4.26 +- 0.99 um
