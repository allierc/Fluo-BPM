"""Are planes near the objective sharper than planes far from it?

They must be. The objective sits past the exit face of the volume, so light from a
plane far from it crosses the whole sample before being collected, while light from
the nearest plane crosses almost nothing. Any refracting sample therefore has to
produce an asymmetric stack: sharp near the objective, degraded away from it. With
dn = 0 the propagation is symmetric and the asymmetry must vanish -- that is the
control, not a failure.

This was visible in the first rendering set (archive/02_cells_first_pass), which
differed from the engine configurations in two ways at once: NA 0.8 vs 0.5, and one
shared emission phase screen vs one per plane. This script separates them.

    python test_depth_sharpness.py

Reports, for each arm, a sharpness profile over depth and the near/far ratio.
Sharpness is normalized gradient energy per plane, mean|grad I|^2 / mean(I)^2, which
is insensitive to how bright a plane happens to be.
"""
import argparse
import itertools
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from fluorescence_bpm import Config, FluorescenceBPM
from make_cell_phantom import make_phantom

DZ = 1.0
NZ = 64


def sharpness_profile(vol_zyx):
    """Normalized gradient energy per plane."""
    out = []
    for plane in vol_zyx:
        p = plane.astype(np.float64)
        gy, gx = np.gradient(p)
        mean = p.mean()
        out.append(((gx ** 2 + gy ** 2).mean() / mean ** 2) if mean > 0 else np.nan)
    return np.array(out)


def run_arm(fluo, dn, na, axial_incoherent, dx, n_iter, device):
    """fluo, dn are [z, y, x]; returns the depth-indexed image volume [z, y, x]."""
    pad = np.zeros((1,) + fluo.shape[1:], dtype=np.float32)
    f = np.concatenate([pad, fluo], axis=0)
    d = np.concatenate([dn[:1], dn], axis=0)

    config = Config(nm=1.33, na=na, dx=dx, lbda=0.532, dz=DZ,
                    z_min=0.0, z_max=f.shape[0] * DZ, device=device, seed=42,
                    axial_incoherent=axial_incoherent)
    model = FluorescenceBPM(
        config,
        dn=torch.tensor(np.moveaxis(d, 0, -1), device=device),
        fluo=torch.tensor(np.moveaxis(f, 0, -1), device=device),
    )
    I = torch.zeros_like(model.dn)
    with torch.no_grad():
        for _ in range(n_iter):
            I += model(0, None if axial_incoherent else model.sample_phase())
    I = torch.flip(I, dims=[2])[:, :, :fluo.shape[0]]
    return np.moveaxis(I.cpu().numpy(), -1, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--n_iter', type=int, default=200)
    p.add_argument('--dx', type=float, default=0.2, help='first pass used 0.2 um')
    p.add_argument('--dn_cell', type=float, default=0.025)
    p.add_argument('--n_cells', type=int, default=200)
    args = p.parse_args()

    # the first rendering set's phantom, unchanged
    fluo, dn, cells = make_phantom(nx=512, ny=512, nz=NZ, dx=args.dx, dz=DZ,
                                   r_min=3.0, r_max=7.0, n_cells=args.n_cells,
                                   dn_cell=args.dn_cell, seed=0)
    zero = np.zeros_like(dn)
    print(f'phantom: {len(cells)} cells, dx={args.dx} um, dn_cell={args.dn_cell}\n')

    print(f"{'arm':44s} {'near/far sharpness':>19s}   profile (near -> far)")
    for na, axial, refracting in itertools.product((0.8, 0.5), (False, True), (True, False)):
        vol = run_arm(fluo, dn if refracting else zero, na, axial,
                      args.dx, args.n_iter, args.device)
        s = sharpness_profile(vol)
        # the objective sits past the last plane, so 'near' is the high-index end
        near = np.nanmedian(s[-NZ // 4:])
        far = np.nanmedian(s[:NZ // 4])
        label = (f'NA {na}, {"per-plane" if axial else "shared"} phase, '
                 f'dn {"on" if refracting else "0"}')
        coarse = s.reshape(8, -1).mean(axis=1)[::-1]      # near -> far
        print(f'{label:44s} {near/far:19.2f}   '
              + ' '.join(f'{v/coarse.max():.2f}' for v in coarse))


if __name__ == '__main__':
    main()
