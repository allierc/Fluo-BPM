# cells_real_like

Matched by eye to a real widefield plane the user supplied: cells spanning about an eighth of the frame, nearly touching, a wide brightness spread with a few near-saturated cells, heavy out-of-focus haze, and visible grain on that haze. Scaled from the reference by pixel count rather than by microns: cells are ~30 px in diameter with ~10 across the frame, which on the 1 um delivered grid of the other configurations means a 320 px field and 24-32 um cells.


- git rev `0d42420`, seed `11`
- simulation grid 640 x 640 x 64 at 0.5 / 1.0 um
- delivered stack [16, 320, 320] at voxel 1.0 x 1.0 x 4.0 um
- 130 objects, fill fraction 21.10%
- NA 0.5, lambda 0.532 um, n 1.33; lateral resolution 0.53 um, axial 5.45 um
- 200 Monte Carlo realizations
- medium `MediumType.NONE`, cell dn 0.02
- Poisson noise on, read noise 3.0 e, peak 306 photons, bright-voxel SNR 7.8
- timing: phantom 0.4 s, simulation 3.1 s, total 5.7 s
