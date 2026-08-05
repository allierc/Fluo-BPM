# cells_dense_aberrated

Hard variant: same 2000-cell volume, but embedded in a smooth random refractive index landscape (dn_rms 0.004, 25 um correlation) and imaged through an aberrated pupil (0.3 waves spherical, 0.15 astigmatism). The PSF then varies with x, y and z, which is the case a shift-invariant reconstruction cannot fit.


- git rev `f6387c0`, seed `42`
- simulation grid 512 x 512 x 256 at 0.5 / 1.0 um
- delivered stack [64, 256, 256] at voxel 1.0 x 1.0 x 4.0 um
- 2000 objects, fill fraction 15.87%
- NA 0.5, lambda 0.532 um, n 1.33; lateral resolution 0.53 um, axial 5.45 um
- 200 Monte Carlo realizations
- medium `MediumType.SMOOTH_RANDOM`, cell dn 0.02
- Poisson noise on, read noise 2.0 e, peak 305 photons, bright-voxel SNR 11.9
- timing: phantom 11.0 s, simulation 10.1 s, total 26.0 s
