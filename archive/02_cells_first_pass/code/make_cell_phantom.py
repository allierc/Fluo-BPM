"""
Generate a synthetic volume filled with spherical cells of varying brightness.

Writes two TIFF volumes in [z, y, x] order (the format FluorescenceBPM expects):
  data/cells_fluorescence.tif  fluorophore concentration
  data/cells_dn.tif            refractive index contrast dn = n_cell - n_medium

Cells are non-overlapping spheres placed by rejection sampling. Each cell gets an
independent lognormal brightness so the emitted intensities differ from cell to cell.
"""
import argparse
import os

import numpy as np
from tifffile import imwrite


def make_phantom(
    nx=512,
    ny=512,
    nz=64,
    dx=0.2,          # lateral pixel size [um]
    dz=1.0,          # axial step [um]
    r_min=3.0,       # cell radius range [um]
    r_max=7.0,
    n_cells=200,     # target number of cells (rejection sampling may place fewer)
    gap=0.5,         # minimum surface-to-surface spacing [um]
    dn_cell=0.025,   # refractive index contrast of a cell vs medium
    edge=0.4,        # membrane softness [um]
    seed=0,
):
    rng = np.random.default_rng(seed)

    Lx, Ly, Lz = nx * dx, ny * dx, nz * dz
    print(f"volume {Lx:.1f} x {Ly:.1f} x {Lz:.1f} um^3  ({nx} x {ny} x {nz} voxels)")

    # --- place non-overlapping spheres -------------------------------------
    cells = []  # (x, y, z, r, brightness) in um
    max_tries = 200 * n_cells
    tries = 0
    while len(cells) < n_cells and tries < max_tries:
        tries += 1
        r = rng.uniform(r_min, r_max)
        # keep cells fully inside the volume so nothing is clipped at the borders
        x = rng.uniform(r, Lx - r)
        y = rng.uniform(r, Ly - r)
        z = rng.uniform(r, Lz - r)
        ok = True
        for (xc, yc, zc, rc, _) in cells:
            d2 = (x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2
            if d2 < (r + rc + gap) ** 2:
                ok = False
                break
        if ok:
            # lognormal brightness -> a few bright cells, many dim ones
            b = float(np.clip(rng.lognormal(mean=0.0, sigma=0.6), 0.08, 4.0))
            cells.append((x, y, z, r, b))
    print(f"placed {len(cells)} cells in {tries} attempts")

    # --- rasterize ---------------------------------------------------------
    fluo = np.zeros((nz, ny, nx), dtype=np.float32)
    dn = np.zeros((nz, ny, nx), dtype=np.float32)

    x_ax = (np.arange(nx) + 0.5) * dx
    y_ax = (np.arange(ny) + 0.5) * dx
    z_ax = (np.arange(nz) + 0.5) * dz

    for (xc, yc, zc, r, b) in cells:
        pad = r + 3 * edge
        i0, i1 = np.searchsorted(x_ax, [xc - pad, xc + pad])
        j0, j1 = np.searchsorted(y_ax, [yc - pad, yc + pad])
        k0, k1 = np.searchsorted(z_ax, [zc - pad, zc + pad])
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            continue

        X = x_ax[i0:i1].reshape(1, 1, -1) - xc
        Y = y_ax[j0:j1].reshape(1, -1, 1) - yc
        Z = z_ax[k0:k1].reshape(-1, 1, 1) - zc
        dist = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

        # soft-edged indicator, ~1 inside, ~0 outside, tanh membrane of width `edge`
        mask = 0.5 * (1.0 - np.tanh((dist - r) / edge))

        sub_f = fluo[k0:k1, j0:j1, i0:i1]
        sub_d = dn[k0:k1, j0:j1, i0:i1]
        np.maximum(sub_f, (b * mask).astype(np.float32), out=sub_f)
        np.maximum(sub_d, (dn_cell * mask).astype(np.float32), out=sub_d)

    return fluo, dn, cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=512)
    p.add_argument('--ny', type=int, default=512)
    p.add_argument('--nz', type=int, default=64)
    p.add_argument('--dx', type=float, default=0.2)
    p.add_argument('--dz', type=float, default=1.0)
    p.add_argument('--n_cells', type=int, default=200)
    p.add_argument('--r_min', type=float, default=3.0)
    p.add_argument('--r_max', type=float, default=7.0)
    p.add_argument('--dn_cell', type=float, default=0.025)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=str, default='./data')
    p.add_argument('--tag', type=str, default='cells')
    args = p.parse_args()

    fluo, dn, cells = make_phantom(
        nx=args.nx, ny=args.ny, nz=args.nz, dx=args.dx, dz=args.dz,
        n_cells=args.n_cells, r_min=args.r_min, r_max=args.r_max,
        dn_cell=args.dn_cell, seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    f_path = f"{args.out_dir}/{args.tag}_fluorescence.tif"
    d_path = f"{args.out_dir}/{args.tag}_dn.tif"
    imwrite(f_path, fluo)
    imwrite(d_path, dn)

    bright = np.array([c[4] for c in cells])
    radii = np.array([c[3] for c in cells])
    print(f"fluorescence: {f_path}  shape {fluo.shape}  max {fluo.max():.2f}")
    print(f"dn:           {d_path}  shape {dn.shape}  max {dn.max():.4f}")
    print(f"brightness  min {bright.min():.2f}  median {np.median(bright):.2f}  max {bright.max():.2f}")
    print(f"radius [um]  min {radii.min():.2f}  max {radii.max():.2f}")

    # cell table, useful as ground truth
    np.savetxt(
        f"{args.out_dir}/{args.tag}_cells.csv",
        np.array(cells),
        delimiter=',',
        header='x_um,y_um,z_um,r_um,brightness',
        comments='',
    )


if __name__ == '__main__':
    main()
