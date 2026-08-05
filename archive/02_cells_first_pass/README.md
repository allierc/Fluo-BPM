# 02 — first spherical-cell volume

The first pass at "a volume filled with cells emitting different intensities",
written directly against the repository's `run_simulation`, before the work was
turned into a configuration-driven engine.

## What it is

200 non-overlapping spheres, radius 3–7 um, in a 102 x 102 x 64 um volume
(512 x 512 x 64 at 0.2 um lateral, 1 um axial). Each cell gets an independent
lognormal brightness, so the emitted intensities differ cell to cell, and a
refractive-index contrast of 0.025 against the medium. NA 0.8, lambda 0.532 um,
200 random-phase realizations.

`fluo_cells_first.gif` is the resulting z-sweep.

## What it showed

- The pipeline works end to end, at ~110 realizations/s for 64 planes of 512^2 on
  one A6000.
- 20 realizations is far too few: the accumulated stack is dominated by speckle
  from the random emission phases. 200 is usable, 1000 is clean.
- 200 cells in a 64 um cube is a 21% fill fraction, dense enough that the
  out-of-focus haze buries the individual cells. The dataset configurations use
  3000 cells in 256^3 um, a 7% fill.

## Why it was replaced

Everything here is a hard-coded argument in a script, and the run is not
reproducible: the Monte Carlo phase came from the unseeded global torch RNG, so
two runs of the same script produced different volumes. Both are fixed in the
engine — `Config.seed` with a model-owned generator, and one YAML per experiment.

## Files

- `code/make_cell_phantom.py` — the phantom generator
- `code/run_cells.py` — the run script
- `code/make_gif.py` — the green z-sweep renderer
