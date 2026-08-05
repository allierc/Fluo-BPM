# cells_real_like — measured against ground truth

- 130 objects; median detected peak / true brightness 71.1 photons per unit
- detection efficiency, surface / deepest octant: 1.14  (no measurable depth trend across this volume)
- brightness recovery: peak vs true brightness r = 0.830, integrated vs true emission r = 0.874
- offset of the brightest voxel from the true centre: lateral median 3.00 um (p90 4.24), axial median 8.0 um (p90 12.0). The axial figure is quantized by the 4 um voxel and grows with cell radius, since the search window spans the cell (+-r) and any plane inside a big cell can be the brightest
- nearest-neighbour distance median 28.4 um; 0.0% of cells closer than the axial resolution (5.5 um)
- noise check: measured std 8.3 vs shot+read prediction 8.3 (ratio 1.00)
