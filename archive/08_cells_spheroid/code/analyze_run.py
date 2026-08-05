"""Measure a delivered dataset against its own ground truth.

A ground-truth dataset is only useful if the properties it claims are measured
rather than asserted. For each log folder this reports:

  detection efficiency vs depth  detected peak / true brightness, per cell. Flat
        when the sample is index matched; falling towards the surface once the cells
        refract, because that light crosses more tissue before the pupil.
  brightness recovery            correlation between true and detected brightness,
        the quantity an intensity-fitting algorithm is trying to recover.
  localization offset            distance from each cell's true centre to the
        nearest local maximum, i.e. how far a perfect detector could be off.
  crowding                       fraction of cells whose nearest neighbour is
        closer than the axial resolution, which is where fitting degenerates.
  measured noise                 std of (noisy - clean) against the shot-noise
        prediction, as a check that the camera model is doing what it claims.

Results go to analysis.md and analysis.json; there is no figure -- the numbers are
the point, and a plot of six scatter panels per run was not being read.

    python analyze_run.py log/cells_dense
    python analyze_run.py log/*            # all of them, plus a comparison table
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from skimage import io

def load_run(folder: Path):
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    clean = io.imread(folder / 'fluo_without_noise.tif').astype(np.float32)
    noisy_path = folder / 'fluo_with_noise.tif'
    noisy = io.imread(noisy_path).astype(np.float32) if noisy_path.exists() else None
    gt = np.loadtxt(folder / 'gt_cells.csv', delimiter=',', skiprows=1)
    if gt.ndim == 1:
        gt = gt[None, :]
    return summary, clean, noisy, gt


def analyze(folder: Path):
    summary, clean, noisy, gt = load_run(folder)
    vx, vy, vz = summary['voxel_um']
    nz, ny, nx = clean.shape

    label, x_um, y_um, z_um, r_um, bright = (gt[:, 0], gt[:, 1], gt[:, 2],
                                            gt[:, 3], gt[:, 4], gt[:, 5])

    # --- per-cell detected peak and integrated signal ----------------------
    # Lateral and axial offsets are kept apart: at a 4 um z voxel a cell spans one to
    # three planes, so the brightest plane is often a neighbour of the one holding the
    # centre. Pooling the two into one distance reports that voxelization as if it
    # were a localization error.
    detected, integrated, off_lat, off_ax = [], [], [], []
    win = 3
    for xi, yi, zi, ri in zip(x_um, y_um, z_um, r_um):
        k = int(np.clip(round(zi / vz - 0.5), 0, nz - 1))
        j = int(np.clip(round(yi / vy - 0.5), 0, ny - 1))
        i = int(np.clip(round(xi / vx - 0.5), 0, nx - 1))
        kw = max(int(round(ri / vz)), 1)
        k0, j0, i0 = max(k - kw, 0), max(j - win, 0), max(i - win, 0)
        sub = clean[k0:k + kw + 1, j0:j + win + 1, i0:i + win + 1]
        detected.append(float(sub.max()))

        # integrated signal over the cell body, the quantity that scales with the
        # total emission an amplitude-fitting method is trying to recover
        rj = max(int(round(ri / vy)), 1)
        ri_ = max(int(round(ri / vx)), 1)
        body = clean[max(k - kw, 0):k + kw + 1,
                     max(j - rj, 0):j + rj + 1,
                     max(i - ri_, 0):i + ri_ + 1]
        integrated.append(float(body.sum()))

        dk, dj, di = np.unravel_index(sub.argmax(), sub.shape)
        off_lat.append(float(np.hypot((j0 + dj - j) * vy, (i0 + di - i) * vx)))
        off_ax.append(float(abs((k0 + dk - k) * vz)))

    detected = np.array(detected)
    integrated = np.array(integrated)
    off_lat, off_ax = np.array(off_lat), np.array(off_ax)
    efficiency = detected / np.clip(bright, 1e-9, None)
    true_emission = bright * (4.0 / 3.0) * np.pi * r_um ** 3

    # --- depth profile, over the bins the phantom actually occupies ---------
    n_bins = 8
    edges = np.linspace(0, nz * vz, n_bins + 1)
    which = np.clip(np.digitize(z_um, edges) - 1, 0, n_bins - 1)
    depth_profile = np.array([np.median(efficiency[which == b]) if (which == b).any()
                              else np.nan for b in range(n_bins)])
    ref = np.nanmax(depth_profile)
    depth_profile_norm = depth_profile / ref if ref > 0 else depth_profile
    occupied = np.flatnonzero(np.isfinite(depth_profile_norm))
    if len(occupied) >= 2:
        surface_vs_deep = float(depth_profile_norm[occupied[0]]
                                / depth_profile_norm[occupied[-1]])
    else:
        surface_vs_deep = float('nan')

    # --- brightness recovery ----------------------------------------------
    def safe_corr(a, b):
        """nan for a constant series is undefined, not a failure: a bead grid has one
        brightness by design, so there is no correlation to report."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        # np.std over identical floats returns ~1e-18 rather than exactly 0, so a
        # constant series has to be caught with a relative tolerance
        def constant(v):
            return v.std() <= 1e-9 * max(abs(float(v.mean())), 1.0)
        if ok.sum() < 3 or constant(a[ok]) or constant(b[ok]):
            return None
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    corr = safe_corr(bright, detected)
    corr_total = safe_corr(true_emission, integrated)

    # --- crowding ----------------------------------------------------------
    axial_res = summary['sampling']['axial_resolution_um']
    pts = np.stack([x_um, y_um, z_um], axis=1)
    if len(pts) > 1:
        # blockwise nearest neighbour, cheap enough for a few thousand cells
        nn = np.full(len(pts), np.inf)
        block = 512
        for start in range(0, len(pts), block):
            chunk = pts[start:start + block]
            d = np.linalg.norm(chunk[:, None, :] - pts[None, :, :], axis=2)
            d[np.arange(len(chunk)), np.arange(start, start + len(chunk))] = np.inf
            nn[start:start + block] = d.min(axis=1)
        crowded = float((nn < axial_res).mean())
        nn_median = float(np.median(nn))
    else:
        crowded, nn_median = 0.0, float('nan')

    # --- noise check -------------------------------------------------------
    noise_check = {}
    if noisy is not None and summary['photons'].get('noise_applied', True):
        cm_offset = 100.0
        residual = (noisy - cm_offset) - clean
        # bin voxels by expected photon count and compare std to sqrt(N)
        lo, hi = np.percentile(clean, [50, 99.9])
        sel = (clean > lo) & (clean < hi)
        measured = float(residual[sel].std())
        predicted = float(np.sqrt(clean[sel].mean() + summary['photons']['read_noise_e'] ** 2))
        noise_check = {'measured_std': measured, 'predicted_std': predicted,
                       'ratio': measured / predicted if predicted > 0 else np.nan}

    result = {
        'name': summary['name'],
        'n_cells': int(len(gt)),
        'efficiency_median': float(np.median(efficiency)),
        'efficiency_depth_profile': [float(v) for v in depth_profile_norm],
        'efficiency_surface_vs_deep': surface_vs_deep,
        'brightness_correlation_peak': corr,
        'brightness_correlation_integrated': corr_total,
        'offset_lateral_um_median': float(np.median(off_lat)),
        'offset_lateral_um_p90': float(np.percentile(off_lat, 90)),
        'offset_axial_um_median': float(np.median(off_ax)),
        'offset_axial_um_p90': float(np.percentile(off_ax, 90)),
        'nearest_neighbour_um_median': nn_median,
        'crowded_fraction': crowded,
        'axial_resolution_um': axial_res,
        'voxel_z_um': vz,
        'noise': noise_check,
    }

    (Path(folder) / 'analysis.json').write_text(json.dumps(result, indent=2))
    _markdown(Path(folder) / 'analysis.md', result)
    return result


def _markdown(path, r):
    lines = [
        f"# {r['name']} — measured against ground truth", '',
        f"- {r['n_cells']} objects; median detected peak / true brightness "
        f"{r['efficiency_median']:.1f} photons per unit",
        f"- detection efficiency, surface / deepest octant: "
        f"{r['efficiency_surface_vs_deep']:.2f}"
        + ("  (flat: index-matched sample)" if r['efficiency_surface_vs_deep'] > 0.9
           else "  (light from shallow cells crosses more tissue)"),
        "- brightness recovery: "
        + ('peak vs true brightness r = %.3f' % r['brightness_correlation_peak']
           if r['brightness_correlation_peak'] is not None
           else 'uniform brightness, nothing to correlate')
        + (', integrated vs true emission r = %.3f' % r['brightness_correlation_integrated']
           if r['brightness_correlation_integrated'] is not None else ''),
        f"- offset of the brightest voxel from the true centre: lateral median "
        f"{r['offset_lateral_um_median']:.2f} um (p90 {r['offset_lateral_um_p90']:.2f}), "
        f"axial median {r['offset_axial_um_median']:.1f} um (p90 "
        f"{r['offset_axial_um_p90']:.1f}); the axial figure is bounded below by the "
        f"{r['voxel_z_um']:.0f} um voxel",
        f"- nearest-neighbour distance median "
        f"{r['nearest_neighbour_um_median']:.1f} um; "
        f"{r['crowded_fraction']*100:.1f}% of cells closer than the axial resolution "
        f"({r['axial_resolution_um']:.1f} um)",
    ]
    if r['noise']:
        n = r['noise']
        lines.append(f"- noise check: measured std {n['measured_std']:.1f} vs shot+read "
                     f"prediction {n['predicted_std']:.1f} (ratio {n['ratio']:.2f})")
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folders', nargs='+')
    args = p.parse_args()

    results = []
    for f in args.folders:
        folder = Path(f)
        if not (folder / 'summary.json').exists():
            continue
        r = analyze(folder)
        results.append(r)
        rt = r['brightness_correlation_integrated']
        print(f"{r['name']:26s} eff(surface/deep) {r['efficiency_surface_vs_deep']:.2f}  "
              f"r(emission) {'  n/a' if rt is None else f'{rt:.3f}'}  "
              f"lateral offset {r['offset_lateral_um_median']:.2f} um  "
              f"nn {r['nearest_neighbour_um_median']:.1f} um  "
              f"crowded {r['crowded_fraction']*100:.1f}%")

    if len(results) > 1:
        print('\n| run | cells | eff surface/deep | r(emission) | lateral offset [um] '
              '| nn [um] | crowded |')
        print('|---|---|---|---|---|---|---|')
        for r in results:
            rt = r['brightness_correlation_integrated']
            print(f"| {r['name']} | {r['n_cells']} | "
                  f"{r['efficiency_surface_vs_deep']:.2f} | "
                  f"{'n/a' if rt is None else f'{rt:.3f}'} | "
                  f"{r['offset_lateral_um_median']:.2f} | "
                  f"{r['nearest_neighbour_um_median']:.1f} | "
                  f"{r['crowded_fraction']*100:.1f}% |")
