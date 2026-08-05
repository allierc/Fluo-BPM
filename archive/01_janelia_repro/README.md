# 01 — reproducing assets/fluo.gif

Attempt to reproduce the animation on the repository front page, as the starting
point for everything that follows.

## What the animation is

29 frames, 512 x 512, a green z-sweep. The frames are not the same structure
defocusing: frame 0 shows the *janelia Research Campus* logo, frame 14 shows
*hhmi*. So it sweeps through a volume with different structures printed at
different depths — the cryolite sample of `load_zarr.py`.

## Why it cannot be reproduced exactly

The source data is not in the repository and is not on this filesystem:

- `.gitignore` excludes `*.tif` and `data/*.tif`, so `data/` holds only its README.
- `cryolite_binary.zarr` / `cryolite_measurement.zarr`, which `load_zarr.py` reads,
  are absent (searched `/workspace` and `/groups/saalfeld/home/allierc`).

**To reproduce it exactly, the two zarr volumes are needed.** Point
`load_zarr.py` at them and the original pipeline runs.

## What was done instead, and what it showed

The published frames were thresholded back into a binary fluorophore volume
(`make_janelia_phantom.py`) and re-simulated with the parameters of
`run_simulation.py`. That run exposed a real property of those parameters:

At `dx = 2.412 um` the Nyquist frequency is `1/(2 dx) = 0.207 um^-1`, while the
pupil cutoff is `NA/lbda = 0.95/0.637 = 1.49 um^-1`. **The pupil mask is seven
times larger than the sampled k-space, so it passes everything and performs no
filtering at all.** Over one `dz = 1.989 um` step the largest defocus phase
anywhere in k-space is 0.13 rad.

Consequence: at that sampling the model has no optical sectioning. Measured on
the re-simulated volume, all 29 output planes correlate with each other at
>= 0.93, and every plane's brightest match is the same source plane. A stack in
which each depth shows its own structure — which is what the animation shows —
cannot come out of this configuration. The animation is therefore either the
measured volume (`cryolite_measurement.zarr`) or a run at finer sampling.

`fluo_repro.gif` is the re-simulation; `fluo_original.gif` is the published
animation. The letters are recognizable but bloomed, consistent with the absent
band limit.

## What carried forward

The engine now refuses this configuration rather than producing a stack that
looks plausible and has no optics in it (`FluoBPMConfig` validator, which reports
the maximum usable NA for a given `dx`). Every dataset configuration in
`config/` samples at `dx = 0.5 um` with NA 0.5, where the pupil sits inside
k-space.

## Files

- `code/make_janelia_phantom.py` — recovers the phantom from the GIF frames
- `code/run_janelia.py` — `--preset cryolite` (published parameters) and
  `--preset optical` (same phantom at microscope sampling)
