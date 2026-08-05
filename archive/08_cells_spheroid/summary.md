# cells_spheroid

Spheroid rather than dispersed cells: 550 larger cells (radius 5-7 um) packed inside one 70 um ball at the centre of the volume. Light reaching the far side has crossed the whole aggregate, so this is the variant where depth-dependent scattering dominates.


- git rev `f6387c0`, seed `7`
- simulation grid 512 x 512 x 256 at 0.5 / 1.0 um
- delivered stack [64, 256, 256] at voxel 1.0 x 1.0 x 4.0 um
- 550 objects, fill fraction 2.54%
- NA 0.5, lambda 0.532 um, n 1.33; lateral resolution 0.53 um, axial 5.45 um
- 200 Monte Carlo realizations
- medium `MediumType.NONE`, cell dn 0.025
- Poisson noise on, read noise 2.0 e, peak 305 photons, bright-voxel SNR 10.6
- timing: phantom 2.4 s, simulation 10.1 s, total 16.7 s
