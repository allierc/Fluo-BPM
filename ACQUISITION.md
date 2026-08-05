# Reference acquisition

Extracted from the `.czi` metadata of
`/groups/saalfeld/saalfeldlab/zapbench-release/confocal_emf3.czi`. Metadata
reliability is unverified — treat every number below as reported by ZEN, not as
measured. Where the simulator departs from it, that is called out in
"What the simulator does with this".

## Optics

| | |
|---|---|
| Objective | W Plan-Apochromat 20x/1,0 DIC (UV)VIS-IR M27 75mm |
| NA | 1.0, water immersion, n = 1.33 |
| Working distance | 1880 µm |
| Microscope | Zeiss Axio Examiner, upright, System = LSM 800 |
| Scan zoom | 0.8 x 0.8, bidirectional frame scan, speed 7 |
| Pixel dwell / frame time | 0.512 µs / 5.03 s |

Sampling — 258.951 nm in x/y, 1.000 µm in z (3.86x anisotropic), 2048 x 2048 x 269
per tile, 2 mosaic tiles stacked in y (origins y=0 and y=1844 px, so 204 px /
52.8 µm overlap), 16-bit, acquired 2022-04-04 in ZEN blue 3.4.

## Channels

| | Ch0 AF488-T1 | Ch1 AF568-T2 |
|---|---|---|
| Dye | Alexa Fluor 488 | Alexa Fluor 568 |
| Ex / Em | 493 / 517 nm | 577 / 603 nm |
| Laser | 488 nm @ 4% | 561 nm @ 4% |
| Detection band | 499.6-535.1 nm | 571.8-755.3 nm |
| Detector | Spectral, GaAsP-PMT | MA-Pmt2, Multialkali-PMT |
| PMT voltage | 750.0 V | 750.0 V |
| Digital gain | 0.3 | 1.0 |
| Offset (DN) | 12.9 | 80 |
| PCF | 247.70 DN/PE | 44.41 DN/PE |
| Pinhole | 75.56 µm = 2.54 AU | 75.56 µm = 1.98 AU |

## What follows from these numbers

**Resolution.** At NA 1.0 in water, `sin(theta) = 1.0/1.33 = 0.75`, a half-angle of
48.7 deg. For Ch0 (517 nm emission): lateral FWHM `0.5*lbda/NA = 0.26 µm`, axial
FWHM `lbda/(n - sqrt(n^2 - NA^2)) = 1.14 µm`.

**The xy pixel is at the band limit, not below it.** 258.951 nm equals
`lbda/(2*NA) = 258.5 nm` to within 0.2%. So the delivered image is sampled exactly
at the diffraction cutoff — critically sampled for intensity in the Nyquist sense
only if one accepts one sample per resolution element; a factor of two finer would
be needed to sample the PSF itself. Any simulation that wants to *represent* this
optical transfer function must run on a finer grid than the delivered pixel: the
engine rejects a configuration where `dx >= lbda/(2*NA)`, because there the pupil
mask is larger than the sampled k-space and stops filtering at all.

**z is coarse relative to the axial PSF.** 1.0 µm steps against a 1.14 µm axial
FWHM: roughly one sample per resolution element axially, 3.86x anisotropic against
xy.

**Photon budget is small, and this sets the noise.** PCF is the photon conversion
factor, DN per photo-electron. Ch0 at 247.70 DN/PE with a 16-bit range means
saturation at `65535/247.70 = 265 photo-electrons`; Ch1 at 44.41 DN/PE saturates at
1476 PE. So a bright voxel in Ch0 carries at most a few hundred detected photons,
and `sqrt(265) / 265 = 6%` relative shot noise at saturation, worse everywhere
else. **The grain in these images is photon noise, not refractive index
structure.** Offsets are 12.9 DN (Ch0) and 80 DN (Ch1).

**Cell size in pixels.** At 258.951 nm, a cell ~30 px across is 7.8 µm — an
ordinary nucleus. Earlier estimates in this repository assumed a 1 µm pixel and so
implied 30 µm cells; that was wrong by a factor of four.

## What the simulator does with this

| quantity | reference | simulation |
|---|---|---|
| NA / n / lambda | 1.0 / 1.33 / 0.517 µm | same |
| delivered xy pixel | 258.951 nm | 259 nm (0.1295 µm grid, `bin_xy: 2`) |
| delivered z step | 1.000 µm | 1.0 µm (`dz: 1.0`, `bin_z: 1`) |
| stack depth | 269 planes | 256 planes |
| field of view | 2048 px = 530 µm | 512 px = 133 µm (a crop; the full tile would need a 4096^2 grid) |
| cell diameter | ~30 px = 7.8 µm | 7-9 µm (radius 3.5-4.5 µm) |
| gain / offset | 247.70 DN/PE, 12.9 DN | same |
| peak signal | <= 265 PE (16-bit saturation) | 200 PE |

**Known deviation: the pinhole is not modelled.** This is a confocal LSM 800 with a
2.54 AU pinhole on Ch0, which rejects part of the out-of-focus light. The engine
models widefield detection: emission from every plane reaches the detector. The
simulated stacks therefore carry *more* out-of-focus haze than the reference, and a
dense phantom looks flatter than the real data. Matching that properly needs a
different forward model — per-plane illumination and a pinhole in the detection path
— not a parameter change.
