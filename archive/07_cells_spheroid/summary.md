# cells_spheroid

Spheroid rather than dispersed cells: 550 larger cells (radius 5-7 um) packed inside one 70 um ball at the centre of the volume. Light reaching the far side has crossed the whole aggregate, so this is the variant where depth-dependent scattering dominates.


- git rev `587705f`, seed `7`
- simulation grid 512 x 512 x 256 at 0.5 / 1.0 um
- delivered stack [64, 256, 256] at voxel 1.0 x 1.0 x 4.0 um
- 550 objects, fill fraction 2.54%
- NA 0.5, lambda 0.532 um, n 1.33; lateral resolution 0.53 um, axial 5.45 um
- 200 Monte Carlo realizations
- medium `MediumType.NONE`, cell dn 0.025
- Poisson noise on, read noise 2.0 e, peak 305 photons, bright-voxel SNR 12.9
- timing: phantom 2.3 s, simulation 11.6 s, total 18.1 s
