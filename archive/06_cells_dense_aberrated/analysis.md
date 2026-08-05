# cells_dense_aberrated — measured against ground truth

- 2000 objects; median detected peak / true brightness 142.8 photons per unit
- detection efficiency, surface / deepest octant: 0.90  (no measurable depth trend across this volume)
- brightness recovery: peak vs true brightness r = 0.438, integrated vs true emission r = 0.671
- offset of the brightest voxel from the true centre: lateral median 3.00 um (p90 4.24), axial median 4.0 um (p90 8.0). The axial figure is quantized by the 4 um voxel and grows with cell radius, since the search window spans the cell (+-r) and any plane inside a big cell can be the brightest
- nearest-neighbour distance median 15.3 um; 0.0% of cells closer than the axial resolution (5.5 um)
- noise check: measured std 11.6 vs shot+read prediction 11.6 (ratio 1.00)
