"""Animations and figures for a run's log folder.

Black background, no titles, labels top-left, matching the convention used for the
other figures in this project.
"""
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def green_frames(vol, gamma=0.6, pct=99.8, per_plane=False):
    """[z, y, x] -> list of RGB frames on the green colormap of assets/fluo.gif."""
    vol = np.nan_to_num(np.asarray(vol, dtype=np.float32))
    vol = np.clip(vol - np.percentile(vol, 1.0), 0, None)
    vmax_global = np.percentile(vol, pct)
    frames = []
    for z in range(vol.shape[0]):
        plane = vol[z]
        vmax = np.percentile(plane, pct) if per_plane else vmax_global
        g = np.clip(plane / (vmax if vmax > 0 else 1.0), 0, 1) ** gamma
        rgb = np.zeros(plane.shape + (3,), dtype=np.uint8)
        rgb[..., 1] = (g * 255).astype(np.uint8)
        rgb[..., 0] = (g ** 3 * 40).astype(np.uint8)
        rgb[..., 2] = (g ** 3 * 40).astype(np.uint8)
        frames.append(Image.fromarray(rgb))
    return frames


def label_frames(frames, texts, colour=(255, 255, 255)):
    """Burn a top-left label into each frame (PIL default font, no external deps)."""
    from PIL import ImageDraw
    out = []
    for frame, text in zip(frames, texts):
        f = frame.copy()
        ImageDraw.Draw(f).text((8, 6), text, fill=colour)
        out.append(f)
    return out


def save_gif(vol, path, gamma=0.6, duration=80, per_plane=False, max_frames=120,
             labels=None, pingpong=True):
    """Green z-sweep GIF. Strides the stack when it is longer than max_frames."""
    vol = np.asarray(vol)
    step = max(1, int(np.ceil(vol.shape[0] / max_frames)))
    idx = list(range(0, vol.shape[0], step))
    frames = green_frames(vol[idx], gamma=gamma, per_plane=per_plane)
    if labels is not None:
        frames = label_frames(frames, [labels[i] for i in idx])
    if pingpong and len(frames) > 2:
        frames = frames + frames[-2:0:-1]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    return len(frames)


def save_projection_figure(clean, noisy, gt, path, dx, dz, label_prefix=''):
    """xy and xz maximum projections of ground truth / clean image / noisy image."""
    def mip(v, axis):
        return np.max(np.asarray(v, dtype=np.float32), axis=axis)

    panels = [
        ('a  ground truth  xy', mip(gt, 0)), ('b  image  xy', mip(clean, 0)),
        ('c  image + noise  xy', mip(noisy, 0)),
        ('d  ground truth  xz', mip(gt, 1)), ('e  image  xz', mip(clean, 1)),
        ('f  image + noise  xz', mip(noisy, 1)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), facecolor='black')
    for ax, (label, img) in zip(axes.ravel(), panels):
        aspect = (dz / dx) if 'xz' in label else 1.0
        ax.imshow(img ** 0.5, cmap='Greens_r', aspect=aspect,
                  vmax=np.percentile(img ** 0.5, 99.8))
        ax.text(0.02, 0.97, f'{label_prefix}{label}', color='white', fontsize=11,
                va='top', ha='left', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('black')
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor='black')
    plt.close(fig)


def save_psf_figure(psf_table, path):
    """PSF width against field position and depth, from the bead fits."""
    z = psf_table['z_um']
    r = np.sqrt((psf_table['x_um'] - psf_table['x_um'].mean()) ** 2
                + (psf_table['y_um'] - psf_table['y_um'].mean()) ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor='black')
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        for s in ax.spines.values():
            s.set_color('white')

    axes[0].scatter(z, psf_table['fwhm_xy_um'], c='#e05a5a', s=14)
    axes[0].set_xlabel('depth z [um]', color='white')
    axes[0].set_ylabel('lateral FWHM [um]', color='white')
    axes[0].text(0.02, 0.97, 'a  lateral width vs depth', color='white', fontsize=11,
                 va='top', transform=axes[0].transAxes)

    axes[1].scatter(z, psf_table['fwhm_z_um'], c='#5a8ce0', s=14)
    axes[1].set_xlabel('depth z [um]', color='white')
    axes[1].set_ylabel('axial FWHM [um]', color='white')
    axes[1].text(0.02, 0.97, 'b  axial width vs depth', color='white', fontsize=11,
                 va='top', transform=axes[1].transAxes)

    sc = axes[2].scatter(r, psf_table['fwhm_xy_um'], c=z, cmap='viridis', s=14)
    axes[2].set_xlabel('field radius [um]', color='white')
    axes[2].set_ylabel('lateral FWHM [um]', color='white')
    axes[2].text(0.02, 0.97, 'c  lateral width vs field position', color='white',
                 fontsize=11, va='top', transform=axes[2].transAxes)
    cb = fig.colorbar(sc, ax=axes[2])
    cb.set_label('z [um]', color='white')
    cb.ax.tick_params(colors='white')

    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor='black')
    plt.close(fig)
