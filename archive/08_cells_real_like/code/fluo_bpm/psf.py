"""Measure the simulated PSF where it actually lands.

The bead phantom puts sub-resolution emitters on an (x, y, z) grid; each one is a
delta, so the local image around it *is* the PSF at that field point and depth.

Widths are half-maximum widths of profiles taken *through the peak*, not second
moments of the patch. Two reasons:

- A widefield PSF has an out-of-focus halo whose energy grows with the axial range
  you include, so a second moment is a function of the patch size, not of the
  optics. Measured at NA 0.5 the moment width came out 2.4 um against a 0.53 um
  diffraction limit, purely from summing +-20 um of defocused light into the
  lateral profile.
- FWHM is the number the resolution formulas quote, so it can be checked:
  lateral 0.5*lbda/NA, axial lbda/(n - sqrt(n^2 - NA^2)).

The second-moment sigmas are still reported alongside, since they are the honest
measure of how much light sits in the tails -- which is what a fitting algorithm
has to contend with.
"""
from typing import Dict, List

import numpy as np
import torch

from .phantom import plane_of_depth


def _half_max_width(profile, coords):
    """Full width at half maximum of a background-subtracted profile through its peak.

    Walks out from the peak to the first half-max crossing on each side and
    interpolates linearly. Returns nan if the profile does not fall to half max
    inside the window (an honest 'wider than the patch').
    """
    p = np.asarray(profile, dtype=np.float64)
    base = np.median(p)
    p = p - base
    if p.max() <= 0:
        return np.nan
    i_peak = int(p.argmax())
    half = 0.5 * p[i_peak]

    def crossing(direction):
        i = i_peak
        while 0 < i < len(p) - 1:
            j = i + direction
            if p[j] <= half:
                # linear interpolation between i and j
                t = (p[i] - half) / max(p[i] - p[j], 1e-12)
                return coords[i] + t * (coords[j] - coords[i])
            i = j
        return np.nan

    left, right = crossing(-1), crossing(+1)
    if np.isnan(left) or np.isnan(right):
        return np.nan
    return float(abs(right - left))


def _moment_sigma(profile, coords):
    p = np.clip(np.asarray(profile, dtype=np.float64) - np.median(profile), 0, None)
    total = p.sum()
    if total <= 0:
        return np.nan
    mean = (p * coords).sum() / total
    return float(np.sqrt(max((p * (coords - mean) ** 2).sum() / total, 0.0)))


def fit_bead_psfs(config, cells: List, I_total: torch.Tensor,
                  patch_um: float = 8.0, patch_z_um: float = 30.0) -> Dict[str, np.ndarray]:
    """Per-bead PSF widths in the un-binned, depth-indexed simulation volume.

    I_total is [y, x, z] -- the axis order the physics core works in (see
    test_registration.py); only nx == ny is assumed nowhere, indices are taken
    from the same transform the engine uses.
    """
    vol = config.volume
    I = I_total.detach().cpu().numpy()
    ny, nx, nz = I.shape

    hx = max(int(round(patch_um / vol.dx)), 3)
    hz = max(int(round(patch_z_um / vol.dz)), 3)

    keys = ('label', 'x_um', 'y_um', 'z_um', 'fwhm_x_um', 'fwhm_y_um', 'fwhm_xy_um',
            'fwhm_z_um', 'sigma_xy_um', 'sigma_z_um', 'peak', 'integrated',
            'halo_fraction')
    out = {k: [] for k in keys}

    for c in cells:
        ix = int(np.clip(round(c.x / vol.dx - 0.5), 0, nx - 1))
        iy = int(np.clip(round(c.y / vol.dx - 0.5), 0, ny - 1))
        iz = plane_of_depth(vol, c.z)

        j0, j1 = max(iy - hx, 0), min(iy + hx + 1, ny)
        i0, i1 = max(ix - hx, 0), min(ix + hx + 1, nx)
        k0, k1 = max(iz - hz, 0), min(iz + hz + 1, nz)
        patch = I[j0:j1, i0:i1, k0:k1]
        if patch.size == 0:
            continue

        # in-focus plane of this bead, located inside the patch rather than assumed
        plane_energy = patch.max(axis=(0, 1))
        kf = int(plane_energy.argmax())
        focal = patch[:, :, kf]
        jf, if_ = np.unravel_index(focal.argmax(), focal.shape)

        cy = (np.arange(j0, j1) + 0.5) * vol.dx
        cx = (np.arange(i0, i1) + 0.5) * vol.dx
        cz = (np.arange(k0, k1) + 1.0) * vol.dz          # plane_depths convention

        prof_x = focal[jf, :]
        prof_y = focal[:, if_]
        prof_z = patch[jf, if_, :]

        fx = _half_max_width(prof_x, cx)
        fy = _half_max_width(prof_y, cy)
        fz = _half_max_width(prof_z, cz)

        peak = float(focal[jf, if_])
        # how much of the light in the patch is outside the in-focus core
        core = float(focal[max(jf - 2, 0):jf + 3, max(if_ - 2, 0):if_ + 3].sum())
        total = float(patch.sum())

        out['label'].append(c.label)
        out['x_um'].append(c.x)
        out['y_um'].append(c.y)
        out['z_um'].append(c.z)
        out['fwhm_x_um'].append(fx)
        out['fwhm_y_um'].append(fy)
        out['fwhm_xy_um'].append(np.nanmean([fx, fy]))
        out['fwhm_z_um'].append(fz)
        out['sigma_xy_um'].append(np.nanmean([_moment_sigma(prof_x, cx),
                                              _moment_sigma(prof_y, cy)]))
        out['sigma_z_um'].append(_moment_sigma(prof_z, cz))
        out['peak'].append(peak)
        out['integrated'].append(total)
        out['halo_fraction'].append(1.0 - core / total if total > 0 else np.nan)

    return {k: np.array(v, dtype=float if k != 'label' else int) for k, v in out.items()}


def write_psf_table(psf: Dict[str, np.ndarray], path):
    keys = list(psf.keys())
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for row in zip(*[psf[k] for k in keys]):
            f.write(','.join(f'{v:.6g}' for v in row) + '\n')
