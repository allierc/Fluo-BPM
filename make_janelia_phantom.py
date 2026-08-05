"""
Rebuild the cryolite ("janelia" / "hhmi" 3D-printed logo) source volume used for
assets/fluo.gif.

The original data (data/cryolite_binary.zarr, see load_zarr.py) is not in the repo,
so the phantom is recovered from the published animation: each GIF frame is one
refocused z-plane, and the printed structure is binary, so thresholding each frame
gives back the fluorophore mask of that plane.

Writes, in [z, y, x] order:
  data/janelia_fluorescence.tif   fluorophore concentration (0 / 175, as in load_zarr.py)
  data/janelia_dn.tif             refractive index contrast (zero, as in load_zarr.py)
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tifffile import imwrite


def frames_from_gif(path):
    im = Image.open(path)
    out = []
    for z in range(im.n_frames):
        im.seek(z)
        out.append(np.array(im.convert('RGB'))[..., 1].astype(np.float32) / 255.0)
    return np.stack(out)  # [z, y, x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gif', type=str, default='./assets/fluo.gif')
    p.add_argument('--thr', type=float, default=0.52, help='threshold on the gamma-coded frame')
    p.add_argument('--smooth', type=float, default=0.8, help='edge smoothing [px]')
    p.add_argument('--amplitude', type=float, default=175.0, help='fluo scale (load_zarr.py uses 175)')
    p.add_argument('--dn_cell', type=float, default=0.0, help='load_zarr.py sets dn = 0')
    p.add_argument('--out_dir', type=str, default='./data')
    p.add_argument('--tag', type=str, default='janelia')
    args = p.parse_args()

    g = frames_from_gif(args.gif)
    print(f"{args.gif}: {g.shape}  background level {np.median(g):.3f}")

    mask = (g > args.thr).astype(np.float32)
    if args.smooth > 0:
        mask = gaussian_filter(mask, sigma=(0, args.smooth, args.smooth))
    fluo = (args.amplitude * mask).astype(np.float32)
    dn = (args.dn_cell * mask).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    imwrite(f"{args.out_dir}/{args.tag}_fluorescence.tif", fluo)
    imwrite(f"{args.out_dir}/{args.tag}_dn.tif", dn)
    print(f"wrote {args.out_dir}/{args.tag}_fluorescence.tif  shape {fluo.shape}  "
          f"fill fraction {(mask > 0.5).mean():.3f}")


if __name__ == '__main__':
    main()
