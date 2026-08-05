"""
Render a z-sweep animation of a simulated volume, in the style of assets/fluo.gif.

  python make_gif.py --input output/cells/fluorescence_final.tif --output assets/fluo_cells.gif
"""
import argparse

import numpy as np
from PIL import Image
from skimage import io


def to_green_frames(vol, gamma=0.6, pct=99.8, per_plane=False):
    """vol: [z, y, x] float -> list of RGB uint8 frames on a green colormap."""
    vol = np.nan_to_num(vol.astype(np.float32))
    vol = np.clip(vol, 0, None)
    frames = []
    vmax_global = np.percentile(vol, pct)
    for z in range(vol.shape[0]):
        plane = vol[z]
        vmax = np.percentile(plane, pct) if per_plane else vmax_global
        if vmax <= 0:
            vmax = 1.0
        g = np.clip(plane / vmax, 0, 1) ** gamma
        rgb = np.zeros(plane.shape + (3,), dtype=np.uint8)
        rgb[..., 1] = (g * 255).astype(np.uint8)          # green channel
        rgb[..., 0] = (g ** 3 * 40).astype(np.uint8)      # slight warm tint in the core
        rgb[..., 2] = (g ** 3 * 40).astype(np.uint8)
        frames.append(Image.fromarray(rgb))
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='./output/cells/fluorescence_final.tif')
    p.add_argument('--output', type=str, default='./assets/fluo_cells.gif')
    p.add_argument('--gamma', type=float, default=0.6)
    p.add_argument('--pct', type=float, default=99.8)
    p.add_argument('--duration', type=int, default=100, help='ms per frame')
    p.add_argument('--per_plane', action='store_true', help='normalize each plane separately')
    p.add_argument('--pingpong', action='store_true', help='sweep down then back up')
    args = p.parse_args()

    vol = io.imread(args.input)
    print(f"{args.input}: shape {vol.shape}  range [{vol.min():.3g}, {vol.max():.3g}]")

    frames = to_green_frames(vol, gamma=args.gamma, pct=args.pct, per_plane=args.per_plane)
    if args.pingpong:
        frames = frames + frames[-2:0:-1]

    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
        optimize=True,
    )
    print(f"wrote {args.output}  ({len(frames)} frames, {frames[0].size[0]}x{frames[0].size[1]})")


if __name__ == '__main__':
    main()
