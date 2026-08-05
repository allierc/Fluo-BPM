# Archive

Runs of the dataset engine, git rev `587705f`. Each folder is self-contained: config, animations, figures, ground-truth table, measured analysis, and the code that produced it.

| archive | cells | voxel [um] | noise | eff surface/deep | r(emission) | notes |
|---|---|---|---|---|---|---|
| [03_beads_psf](03_beads_psf/) | 325 | 1x1x4 | shot+read | 1.00 | n/a | PSF probe, ideal case. 5 x 5 x 13 sub-resolution beads on a grid throu |
| [04_beads_psf_aberrated](04_beads_psf_aberrated/) | 325 | 1x1x4 | shot+read | 0.46 | n/a | PSF probe, xyz-varying case. 5 x 5 x 13 sub-resolution beads on a grid |
| [05_cells_dense](05_cells_dense/) | 2000 | 1x1x4 | shot+read | 0.87 | 0.679 | Ground-truth dataset, main variant. 2000 spherical cells (radius 5-9 u |
| [06_cells_dense_aberrated](06_cells_dense_aberrated/) | 2000 | 1x1x4 | shot+read | 0.90 | 0.671 | Hard variant: same 2000-cell volume, but embedded in a smooth random r |
| [07_cells_spheroid](07_cells_spheroid/) | 550 | 1x1x4 | shot+read | 0.86 | 0.529 | Spheroid rather than dispersed cells: 550 larger cells (radius 5-7 um) |
| [08_cells_real_like](08_cells_real_like/) | 130 | 1x1x4 | shot+read | 1.14 | 0.874 | Matched by eye to a real widefield plane the user supplied: cells span |
