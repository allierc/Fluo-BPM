# cells_spheroid — measured against ground truth

- 550 objects; median detected peak / true brightness 179.5 photons per unit
- detection efficiency, surface / deepest octant: 0.86  (light from shallow cells crosses more refracting tissue)
- brightness recovery: peak vs true brightness r = 0.402, integrated vs true emission r = 0.529
- offset of the brightest voxel from the true centre: lateral median 2.83 um (p90 3.61), axial median 4.0 um (p90 8.0). The axial figure is quantized by the 4 um voxel and grows with cell radius, since the search window spans the cell (+-r) and any plane inside a big cell can be the brightest
- nearest-neighbour distance median 11.8 um; 0.0% of cells closer than the axial resolution (5.5 um)
- noise check: measured std 7.9 vs shot+read prediction 7.8 (ratio 1.00)
