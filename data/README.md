The xy spacing is 2.412 microns, the z spacing (thickness per plane) is 1.989 microns, and the wavelength is 0.637 microns (637 nm).

The cryolite_likelihood.zarr volume is a binarized float32 volume set to 1.0 where there is cryolite and 0.0 where there is not. You'll want to take 1.0 - cryolite_likelihood to get an idea of where the actual fluorescence is, because the cryolite is inverse of the fluorescence.

The measurement.zarr volume contains the measured intensity as a float32 volume (z y x order).
