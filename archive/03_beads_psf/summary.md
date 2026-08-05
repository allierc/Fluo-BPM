# beads_psf

PSF probe, ideal case. 5 x 5 x 13 sub-resolution beads on a grid through an index-matched volume with an unaberrated pupil: the PSF should depend on depth only through defocus, and not on x or y. This is the reference the aberrated probe is measured against.


- git rev `587705f`, seed `3`
- simulation grid 512 x 512 x 256 at 0.5 / 1.0 um
- delivered stack [64, 256, 256] at voxel 1.0 x 1.0 x 4.0 um
- 325 objects, fill fraction 0.00%
- NA 0.5, lambda 0.532 um, n 1.33; lateral resolution 0.53 um, axial 5.45 um
- 400 Monte Carlo realizations
- medium `MediumType.NONE`, cell dn 0.0
- Poisson noise on, read noise 2.0 e, peak 305 photons, bright-voxel SNR 3.0
- timing: phantom 0.2 s, simulation 22.7 s, total 27.4 s
- PSF over the bead grid: lateral FWHM 0.53 +- 0.00 um, axial 4.70 +- 0.01 um
