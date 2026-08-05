# cells_dense_noisefree

Ground-truth dataset, main variant, NOISE OFF. 2000 spherical cells (radius 5-9 um, so 10-18 um across) with lognormal brightness in a 256 x 256 x 256 um volume, imaged at NA 0.5 and delivered on a 1 x 1 x 4 um voxel grid with shot and read noise. Cells refract (dn = 0.02) so the PSF degrades with depth; the medium is index-matched.


- git rev `f6387c0`, seed `42`
- simulation grid 512 x 512 x 256 at 0.5 / 1.0 um
- delivered stack [64, 256, 256] at voxel 1.0 x 1.0 x 4.0 um
- 2000 objects, fill fraction 15.87%
- NA 0.5, lambda 0.532 um, n 1.33; lateral resolution 0.53 um, axial 5.45 um
- 200 Monte Carlo realizations
- medium `MediumType.NONE`, cell dn 0.02
- Poisson noise off, read noise 0.0 e, peak 305 photons, no detector noise
- timing: phantom 1.1 s, simulation 10.3 s, total 16.1 s
