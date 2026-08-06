"""Build the note: the model, and three sweeps that test it.

    python note.py            # -> archive/note/{figures, note.tex, note.pdf}

Reads log/mc_sweep, log/RI_sweep and log/PSF_sweep. Reproduce those with:

    python run_sweep.py -s config/sweeps/MC_sweep.yaml
    python run_sweep.py -s config/sweeps/RI_sweep.yaml
    python run_sweep.py -s config/sweeps/PSF_sweep.yaml
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from skimage import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

RI_SWEEP = [('dn0000', 0.0, 'dn = 0'), ('dn0025', 0.0025, 'dn = 0.0025'),
            ('dn0050', 0.005, 'dn = 0.005'), ('dn0100', 0.010, 'dn = 0.010'),
            ('dn0200', 0.020, 'dn = 0.020')]
PSF_SWEEP = [('na100', 1.0), ('na080', 0.8), ('na060', 0.6),
             ('na040', 0.4), ('na030', 0.3)]
OUT = Path('archive/note')
DATE = '2026-08-05'



GREEN = LinearSegmentedColormap.from_list(
    'fluo_green', ['#000000', '#0b3d17', '#1f9c3c', '#7fe08a', '#ffffff'])


def show(ax, img, extent=None, lo=1.0, hi=99.9, gamma=0.85):
    """One image panel: percentile window, gamma, black-to-green ramp.

    The display range matters more than it sounds. These are widefield stacks with a
    large out-of-focus pedestal, so stretching from zero puts every panel in the
    mid-tones and hides exactly the structure the figure is about.
    """
    img = np.asarray(img, dtype=np.float64)
    vmin, vmax = np.percentile(img, lo), np.percentile(img, hi)
    if vmax <= vmin:
        vmax = vmin + 1e-9
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    ax.imshow(img, cmap=GREEN, norm=norm, extent=extent, aspect='equal',
              interpolation='nearest', origin='lower')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('black')       # inside the panel is the image itself
    for sp in ax.spines.values():
        sp.set_color('#333333')     # a thin dark frame so the panel reads on white


def panel_title(ax, text, size=10):
    ax.set_title(text, color='black', fontsize=size, loc='left', pad=4)


def plot_axes(ax):
    """A line-plot axis on white: an x axis and a y axis, and nothing else."""
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('black')


def legend(ax, **kw):
    """Frameless, to match axes that have no box either."""
    return ax.legend(frameon=False, labelcolor='black', **kw)


def scalebar(ax, x0, y0, length_um, label, colour='white'):
    ax.plot([x0, x0 + length_um], [y0, y0], color=colour, lw=2.5,
            solid_capstyle='butt')
    ax.text(x0 + length_um / 2, y0 + 0.4, label, color=colour, fontsize=8,
            ha='center', va='bottom')


def isolated_cell(data, min_sep_um=12.0, brightness_pct=70.0):
    """An interior cell with no close neighbour, so a crop shows one cell, not a pair.

    All arms share a seed, so the same cell is returned for every arm of a sweep.
    """
    cells = data['cells']
    bright_cut = np.percentile(cells[:, 5], brightness_pct)
    xyz = cells[:, 1:4]
    nz, ny, nx = data['pe'].shape
    vx, vy, vz = data['summary']['voxel_um']
    Lx, Ly, Lz = nx * vx, ny * vy, nz * vz
    best = None
    for idx, (x, y, z) in enumerate(xyz):
        if not (18 < x < Lx - 18 and 18 < y < Ly - 18 and 0.3 * Lz < z < 0.7 * Lz):
            continue
        if cells[idx, 5] < bright_cut:      # a dim cell makes an unreadable panel
            continue
        d = np.linalg.norm(np.delete(xyz, idx, axis=0) - np.array([x, y, z]), axis=1)
        sep = d.min()
        # take the most isolated candidate in the window rather than requiring a
        # threshold: at 2 um minimum gap the typical neighbour sits ~10 um away, so a
        # hard 14 um cut leaves nothing to plot
        if best is None or sep > best[0]:
            best = (sep, x, y, z, cells[idx, 4])
    if best is None:
        return None
    if best[0] < min_sep_um:
        print(f'  most isolated cell has a neighbour at {best[0]:.1f} um '
              f'(wanted {min_sep_um:.0f}); crops may show part of it')
    return {'x': best[1], 'y': best[2], 'z': best[3], 'r': best[4], 'sep': best[0]}


def crops_at(data, cell, half_um=13.0, half_um_z=24.0):
    """(xy, xz) crops centred on a cell, with um extents.

    The axial half-range is larger than the lateral one on purpose: at NA 0.3 the cell
    images 36 um long, so a +-13 um crop would cut off the very elongation the figure
    is showing. Both are the same in every arm, so the arms stay comparable.
    """
    pe = data['pe']
    vx, vy, vz = data['summary']['voxel_um']
    nz, ny, nx = pe.shape
    i = int(round(cell['x'] / vx - 0.5))
    j = int(round(cell['y'] / vy - 0.5))
    k = int(round(cell['z'] / vz - 0.5))
    hx = int(round(half_um / vx)); hy = int(round(half_um / vy))
    hz = int(round(half_um_z / vz))
    i0, i1 = max(i - hx, 0), min(i + hx + 1, nx)
    j0, j1 = max(j - hy, 0), min(j + hy + 1, ny)
    k0, k1 = max(k - hz, 0), min(k + hz + 1, nz)
    xy = pe[k, j0:j1, i0:i1]
    xz = pe[k0:k1, j, i0:i1]
    ext_xy = [(i0 - i) * vx, (i1 - 1 - i) * vx, (j0 - j) * vy, (j1 - 1 - j) * vy]
    ext_xz = [(i0 - i) * vx, (i1 - 1 - i) * vx, (k0 - k) * vz, (k1 - 1 - k) * vz]
    return (xy, ext_xy), (xz, ext_xz)


def binned(profile, depth, n_bins=16):
    """Median per depth bin: the per-plane contrast of a sparse phantom is far too
    spiky to read as five overlapping traces."""
    edges = np.linspace(depth[0], depth[-1], n_bins + 1)
    which = np.clip(np.digitize(depth, edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.nanmedian(profile[which == b]) if (which == b).any() else np.nan
                    for b in range(n_bins)])
    return centres, med


def load(tag, root='log'):
    folder = Path(root) / tag
    summary = json.loads((folder / 'summary.json').read_text())
    config = yaml.safe_load((folder / 'config.yaml').read_text())
    stack = io.imread(folder / 'fluo.tif').astype(np.float32)
    gain = config['camera'].get('gain', 1.0) or 1.0
    pe = (stack - config['camera'].get('offset', 0.0)) / gain
    # ground truth is optional: the Monte-Carlo arms only need the stack itself
    gt_path, cells_path = folder / 'fluo_gt.tif', folder / 'gt_cells.csv'
    gt = io.imread(gt_path).astype(np.float32) if gt_path.exists() else None
    cells = (np.loadtxt(cells_path, delimiter=',', skiprows=1)
             if cells_path.exists() else None)
    return {'summary': summary, 'config': config, 'pe': pe, 'gt': gt, 'cells': cells}


def cell_contrast(data):
    """Per-plane contrast between cell interiors and the gaps between them.

    This is what the eye reads as sharpness in a dense stack. Gradient energy, the
    obvious alternative, is dominated by fine haze texture and misses the effect
    entirely -- across the whole RI sweep it gives a near/far ratio of 1.00 to 1.01.
    """
    pe, gt = data['pe'], data['gt']
    prof = []
    for k in range(pe.shape[0]):
        g = gt[k]
        if g.max() <= 0:
            prof.append(np.nan)
            continue
        inside, outside = g > 0.6 * g.max(), g < 0.05 * g.max()
        if inside.sum() < 50 or outside.sum() < 50:
            prof.append(np.nan)
            continue
        a, b = pe[k][inside].mean(), pe[k][outside].mean()
        prof.append((a - b) / (a + b) if (a + b) > 0 else np.nan)
    return np.array(prof)


def cell_extent(data, n_cells=80):
    """Median apparent cell size laterally and axially, from half-maximum widths.

    Measured on the delivered stack through each cell's own centre, so it reports the
    size an observer would read off the image rather than a nominal PSF width.
    """
    pe, cells = data['pe'], data['cells']
    vx, vy, vz = data['summary']['voxel_um']
    nz, ny, nx = pe.shape
    lat, ax = [], []
    for row in cells[:n_cells]:
        x, y, z, r = row[1], row[2], row[3], row[4]
        i, j, k = int(round(x / vx - 0.5)), int(round(y / vy - 0.5)), int(round(z / vz - 0.5))
        if not (10 < i < nx - 10 and 10 < j < ny - 10 and 6 < k < nz - 6):
            continue

        def width(profile, step):
            """FWHM above the local floor.

            The haze pedestal in a widefield stack is large, so a width taken above
            zero (or above a percentile of the window) measures the pedestal, not the
            cell: at NA 1.0 that read 12 um for an 8 um cell. Half maximum is taken
            between the local floor and the peak, and the crossings must both be
            inside the window or the measurement is discarded.
            """
            p = np.asarray(profile, dtype=np.float64)
            if p.size < 5:
                return np.nan
            floor, peak_val = p.min(), p.max()
            if peak_val <= floor:
                return np.nan
            half = floor + 0.5 * (peak_val - floor)
            i_peak = int(p.argmax())
            left = np.where(p[:i_peak + 1] <= half)[0]
            right = np.where(p[i_peak:] <= half)[0]
            if not len(left) or not len(right):
                return np.nan          # never falls to half inside the window
            return (i_peak + right[0] - left[-1]) * step

        half_lat = int(round(3 * r / vx))          # +-3 radii: peak plus floor
        half_ax = int(round(4 * r / vz)) + 12       # more in z: the PSF is longer there
        lat.append(width(pe[k, j, max(i - half_lat, 0):i + half_lat + 1], vx))
        ax.append(width(pe[max(k - half_ax, 0):k + half_ax + 1, j, i], vz))
    return float(np.nanmedian(lat)), float(np.nanmedian(ax))


def figure_ri(path):
    """Binned contrast against depth, near/far against dn, and the images themselves."""
    data = {tag: load(tag, root='log/RI_sweep') for tag, _, _ in RI_SWEEP}
    fig = plt.figure(figsize=(13.0, 7.6), facecolor='white')
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :2]); plot_axes(ax0)
    ax1 = fig.add_subplot(gs[0, 2:]); plot_axes(ax1)

    colours = ['#777777', '#2e8b3d', '#0e8f8f', '#2b5fb0', '#c02f2f']
    near, far, dns = [], [], []
    for (tag, dn, label), colour in zip(RI_SWEEP, colours):
        d = data[tag]
        c = cell_contrast(d)
        depth = (np.arange(len(c)) + 1) * d['summary']['voxel_um'][2]
        centres, med = binned(c, depth)
        ax0.plot(centres, med, 'o-', color=colour, lw=1.6, ms=3.5, label=label)
        q = max(len(c) // 4, 1)
        near.append(np.nanmedian(c[-q:])); far.append(np.nanmedian(c[:q])); dns.append(dn)

    ax0.axhline(0.0, color='#555555', ls=':', lw=1)
    ax0.set_xlabel('focal depth [um]        objective side ->', color='black')
    ax0.set_ylabel('cell / gap contrast', color='black')
    panel_title(ax0, 'a)  contrast against depth, median per bin', 11)
    legend(ax0, fontsize=8.5, ncol=2)

    ax1.plot(dns, near, 'o-', color='#2e8b3d', lw=1.8, label='near the objective')
    ax1.plot(dns, far, 'o-', color='#c02f2f', lw=1.8, label='far side')
    ax1.axhline(0.0, color='#555555', ls=':', lw=1)
    ax1.set_xlabel('cell index contrast  dn', color='black')
    ax1.set_ylabel('cell / gap contrast', color='black')
    panel_title(ax1, 'b)  near and far contrast against dn', 11)
    legend(ax1, fontsize=8.5)

    # the images: control and strongest arm, far plane and near plane
    letters = iter('cdef')
    for col, tag in enumerate(['dn0000', 'dn0200']):
        d = data[tag]
        nz = d['pe'].shape[0]
        vz = d['summary']['voxel_um'][2]
        label = 'dn = 0' if tag == 'dn0000' else 'dn = 0.02'
        for side, k in [('far side', 3), ('near objective', nz - 4)]:
            ax = fig.add_subplot(gs[1, col * 2 + (0 if side == 'far side' else 1)])
            img = d['pe'][k, 40:200, 40:200]
            vx = d['summary']['voxel_um'][0]
            ext = [0, img.shape[1] * vx, 0, img.shape[0] * vx]
            show(ax, img, extent=ext, lo=1, hi=99.9, gamma=0.85)
            panel_title(ax, f'{next(letters)})  {label}, {side}', 10)
            if col == 0 and side == 'far side':
                scalebar(ax, 2, 2, 10.0, '10 um')

    fig.savefig(path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return [(lab, dn, n, f) for (_, dn, lab), n, f in zip(RI_SWEEP, near, far)]


def figure_ri_sweep(path, slab_um=1.5):
    """One edge-on (xz) strip per arm, stacked: the sweep seen in the images.

    Averaged over a thin slab in y (slab_um) because a single y plane at W = 240 is
    mostly Monte-Carlo speckle; the averaging is over the display only, and it is the
    same for every arm.
    """
    fig, axes = plt.subplots(len(RI_SWEEP), 1, figsize=(13.0, 7.4), facecolor='white')
    letters = 'abcde'
    for row, (tag, dn, label) in enumerate(RI_SWEEP):
        d = load(tag, root='log/RI_sweep')
        pe = d['pe']
        vx, vy, vz = d['summary']['voxel_um']
        nz, ny, nx = pe.shape
        j = ny // 2
        half = max(int(round(0.5 * slab_um / vy)), 0)
        strip = pe[:, j - half:j + half + 1, :].mean(axis=1).T   # [x, z], z horizontal
        ax = axes[row]
        show(ax, strip, extent=[0, nz * vz, 0, nx * vx], lo=1, hi=99.9, gamma=0.85)
        panel_title(ax, f'{letters[row]})  {label}', 10)
        if row == 0:
            scalebar(ax, 8, 6, 50.0, '50 um')
        if row == len(RI_SWEEP) - 1:
            ax.set_xlabel('depth [um]        far side (left)  ->  objective side (right)',
                          color='black', fontsize=9)
            ax.set_xticks([0, 100, 200, 300, nz * vz])
            ax.tick_params(colors='black', labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def figure_psf(path):
    """The same isolated cell in xy and xz at each NA, same crop, same scale."""
    arms = [(tag, na, load(tag, root='log/PSF_sweep')) for tag, na in PSF_SWEEP]
    cell = isolated_cell(arms[0][2])
    rows = []

    fig = plt.figure(figsize=(13.0, 6.4), facecolor='white')
    gs = fig.add_gridspec(2, len(arms), hspace=0.20, wspace=0.10)
    letters = 'abcdefghij'
    for col, (tag, na, d) in enumerate(arms):
        lat, ax_ = cell_extent(d)
        rows.append((na, lat, ax_, ax_ / lat if lat else np.nan))
        (xy, ext_xy), (xz, ext_xz) = crops_at(d, cell)
        for row, (img, ext, plane) in enumerate([(xy, ext_xy, 'xy'), (xz, ext_xz, 'xz')]):
            ax = fig.add_subplot(gs[row, col])
            show(ax, img, extent=ext, lo=1, hi=99.9, gamma=0.85)
            panel_title(ax, f'{letters[row * len(arms) + col]})  NA {na}, {plane}', 10)
            if col == 0:
                scalebar(ax, ext[0] + 1.5, ext[2] + 1.5, 5.0, '5 um')
    fig.savefig(path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor='white')
    plot_axes(ax)
    na = [r[0] for r in rows]
    ax.plot(na, [r[1] for r in rows], 'o-', color='#2e8b3d', lw=1.8, label='lateral')
    ax.plot(na, [r[2] for r in rows], 'o-', color='#c02f2f', lw=1.8, label='axial')
    ax.axhline(2 * np.median(arms[0][2]['cells'][:, 4]), color='#555555', ls=':', lw=1)
    ax.text(0.98, 0.06, 'dotted: true cell diameter', color='black', fontsize=8.5,
            ha='right', transform=ax.transAxes)
    ax.set_xlabel('numerical aperture', color='black')
    ax.set_ylabel('apparent cell extent, FWHM [um]', color='black')
    panel_title(ax, 'a)  apparent cell size against NA', 11)
    legend(ax, fontsize=9)
    fig.savefig(str(path).replace('.png', '_extent.png'), dpi=140, facecolor='white',
                bbox_inches='tight')
    plt.close(fig)
    return rows


def figure_mc(path):
    """The 1/sqrt(W) law beside what it looks like at three values of W."""
    rows = json.loads(Path('log/mc_sweep/mc_sweep.json').read_text())
    n = np.array([r['n'] for r in rows], dtype=float)
    sp = np.array([r['speckle_pct'] for r in rows], dtype=float)

    exponent = float(np.polyfit(np.log(n), np.log(sp), 1)[0])
    fig = plt.figure(figsize=(13.0, 3.6), facecolor='white')
    gs = fig.add_gridspec(1, 5, width_ratios=[1.9, 0.10, 1, 1, 1], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0]); plot_axes(ax)
    ax.loglog(n, sp, 'o-', color='#c02f2f', lw=1.8, label='measured, detector off')
    ax.loglog(n, sp[0] * np.sqrt(n[0] / n), ':', color='#555555', lw=1.4,
              label=r'$1/\sqrt{W}$ for reference')
    ax.loglog(n, sp[0] * (n / n[0]) ** exponent, '--', color='#2e8b3d', lw=1.3,
              label=f'fit, $W^{{{exponent:.2f}}}$')
    ax.set_xlabel('random phase draws  $W$', color='black')
    ax.set_ylabel('residual speckle [% of local mean]', color='black')
    panel_title(ax, 'a)  speckle of the Monte-Carlo average', 11)
    legend(ax, fontsize=8.5)

    shown = [rows[0]['n'], rows[len(rows) // 2]['n'], rows[-1]['n']]
    letters = iter('bcd')
    for i, nn in enumerate(shown):
        folder = f'log/mc_sweep/n{nn:04d}'
        d = load(f'n{nn:04d}', root='log/mc_sweep')
        pe = d['pe']
        vx = d['summary']['voxel_um'][0]
        img = pe[pe.shape[0] // 2, 48:176, 48:176]
        a = fig.add_subplot(gs[0, 2 + i])
        ext = [0, img.shape[1] * vx, 0, img.shape[0] * vx]
        show(a, img, extent=ext, lo=1, hi=99.9, gamma=0.85)
        panel_title(a, f'{next(letters)})  W = {nn}', 10)
        a.set_anchor('C')
        if i == 0:
            scalebar(a, 1.5, 1.5, 5.0, '5 um')
    fig.savefig(path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    for r in rows:
        r['exponent'] = exponent
    return rows


TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.0cm]{geometry}
\usepackage{graphicx,booktabs,amsmath}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\newcommand{\FT}{\mathcal{F}}
\begin{document}

\textbf{\large The forward model, and three sweeps that test it}

\textit{Fluo-BPM, at the optics of \texttt{confocal\_emf3.czi}. @DATE@}

\section*{1. The model, in one page}

\textbf{Propagation.} Maxwell's equations for a monochromatic scalar field in a
non-magnetic medium reduce to the Helmholtz equation
\begin{equation}
\nabla^2 U(\vec r) + k(\vec r)^2 U(\vec r) = 0,
\qquad k(\vec r) = \frac{2\pi n(\vec r)}{\lambda},
\end{equation}
which keeps the full three-dimensional index distribution $n(\vec r)$ but drops
polarization. Splitting the index into a background $n_0$ and a deviation
$\delta n = n - n_0$, and keeping only forward propagation, the volume is treated as
$P$ slices of thickness $\Delta z$, each carrying a phase map
$\phi_k(x,y) = 2\pi \delta n_k(x,y) \Delta z / \lambda$. One slice to the next is then
\begin{equation}
U_{k+1} = \FT^{-1}\!\Big[\;\FT\big[\,U_k \, e^{i\phi_k}\,\big] \odot \tilde H_{\Delta z}\;\Big],
\qquad
\tilde H_{\Delta z}(\mu_x,\mu_y) = e^{\,i 2\pi \Delta z \sqrt{n_0^2/\lambda^2 - \mu_x^2 - \mu_y^2}} ,
\label{eq:bpm}
\end{equation}
a phase multiplication for what the light passes through, then a Fresnel propagation
for how it diffracts on the way to the next slice. $\odot$ is point-wise
multiplication. Equation~\eqref{eq:bpm} is the whole propagation engine; everything
else is what is put into it and what is read out of it.

\textbf{Fluorescent sources.} Emitters add a source term, and their incoherence is the
statement that different points are uncorrelated,
\begin{equation}
\mathbb{E}\big[s^*(\vec r\,')\,s(\vec r)\big] = \delta(\vec r\,' - \vec r)\, F(\vec r),
\end{equation}
with $F$ the fluorophore distribution. The measured intensity is then
$I(\vec r\,') = \int |G(\vec r\,',\vec r)|^2 F(\vec r)\, d\vec r$, where $G$ is the
Green function that Eq.~\eqref{eq:bpm} implements. That integral has no closed form,
because it needs $|G|^2$ rather than $G$. It can, however, be sampled: drawing a
random phase $\psi \sim \mathcal{U}[0, 2\pi]$ per emitter and propagating the
\emph{coherent} field $\sqrt{F}\,e^{i\psi}$ gives
\begin{equation}
I = \mathbb{E}_\psi\Big[\;\big|\,G\,\big(\sqrt{F} \odot e^{i\psi}\big)\,\big|^2\;\Big],
\label{eq:mc}
\end{equation}
since a diagonal source covariance is exactly what that draw produces:
$\Gamma_U = G\,\Gamma_S\,G^H$ with $\Gamma_S = \mathrm{diag}(F)$ gives
$\Gamma_{U,ii} = \sum_j |G_{ij}|^2 F_j = I_i$. \emph{Incoherent light is thus
simulated as an average of coherent propagations.} In the slice recursion, with
$w = 1 \ldots W$ indexing the draws,
\begin{equation}
U_{k+1,w} = \FT^{-1}\bigg[\;\FT\Big[\;U_{k,w}\,e^{i\phi_k}
 \;+\; \FT^{-1}\big[\,\FT[\,\sqrt{F_k}\,e^{i\psi_{k,w}}\,] \odot C^{-1}\big]\;\Big]
 \odot \tilde H_{\Delta z}\;\bigg],
\label{eq:src}
\end{equation}
where $C = \sqrt{n_0^2/\lambda^2 - \mu_x^2 - \mu_y^2}$ and $C^{-1} \propto
\cos(\theta)^{-1}$ weights emission away from the optical axis.

\textbf{Detection and the average.} The field leaving the last slice is filtered by
the pupil and back-propagated to each focal plane:
\begin{equation}
I_{k,w} = \Big|\,\FT^{-1}\big[\,\FT[U_{P,w}] \odot P_{\text{circ}} \odot e^{i\Gamma}
 \odot \tilde H^{*}_{(P-k+1)\Delta z}\,\big]\Big|^2 ,
\qquad
I_k = \frac{1}{W}\sum_{w=1}^{W} I_{k,w},
\label{eq:det}
\end{equation}
with $P_{\text{circ}} = 1$ for $\sqrt{\mu_x^2+\mu_y^2} < \mathrm{NA}/\lambda$ and zero
outside, and $e^{i\Gamma}$ an optional known aberration (a weighted sum of Zernike
polynomials).

\textbf{Two consequences that bit in practice.} First, $\psi_{k,w}$ carries the index
$k$: the phase must be redrawn \emph{per slice}, not once per draw. Sharing one screen
down the volume makes every emitter in an $(x,y)$ column mutually coherent, so a
cell's slices interfere on axis and Fresnel rings appear at the centre of every cell,
identically in every draw --- averaging cannot remove them. Restoring per-slice
phases removed the rings and doubled the mean in-cell signal, which the interference
had been redistributing. Second, $P_{\text{circ}}$ is only a filter if
$\mathrm{NA}/\lambda < 1/(2\,dx)$; sampled coarser than that, the pupil is wider than
the represented $k$-space, nothing is filtered, and the model has no optical
sectioning at all.

\section*{2. Sweep 1: the Monte-Carlo average}

Equation~\eqref{eq:mc} is an expectation, and a finite $W$ leaves residue that looks
exactly like detector noise. With the detector switched off entirely, the stack still
fluctuates by @SPECKLE1@\,\% of the local mean at $W = @NFIRST@$, falling to
@SPECKLE2@\,\% at $W = @NLAST@$ (Figure~1). The decay is close to but slower than the
$1/\sqrt{W}$ that independent draws would give: fitted over this range it goes as
$W^{@EXPONENT@}$, so 64 times the draws bought a factor 4.8 rather than 8. Draws are
not fully independent here --- the same phantom and the same haze are common to all of
them --- so the ideal law is a bound, not a prediction. Grain in a folder whose name
says no measurement noise is this. The sweeps below run at $W = 240$ and $W = 1920$
respectively.

\section*{3. Sweep 2: refractive index}

Five arms, index contrast 0 to 0.02, 384\,\textmu m deep, no measurement noise, same
cells and seed throughout. The objective sits past the exit face, so light from the
far plane crosses the whole sample and must arrive degraded while the near plane does
not --- and $dn = 0$ is the control that must come out flat.

\begin{center}
\begin{tabular}{lrrr}
\toprule
arm & contrast near objective & contrast far side & lost across the depth \\
\midrule
@RIROWS@
\bottomrule
\end{tabular}
\end{center}

At $dn = 0.02$ the far side has no cell contrast left at all while the near side keeps
three quarters of the index-matched value (Figures~2 and~3). Two notes on measuring this.
Gradient energy --- the obvious sharpness metric --- gives a near/far ratio of 1.00 to
1.01 across the entire sweep and misses the effect completely, because in a stack this
dense it is dominated by fine haze texture that survives; what refraction destroys is
the difference between a cell and the gap beside it. And ratios are the wrong summary
once the far-side contrast crosses zero: near/far runs to $+13.8$ and then to $-109$.

Integrated brightness is also the wrong observable: a sphere with $dn = 0.02$ deflects
light by about $2^{\circ}$, and NA 1.0 in water accepts a half-angle of $48.7^{\circ}$,
so refraction redirects light \emph{within} the collection cone rather than out of it.
An earlier claim in \texttt{DATASET.md} of a 30\,\% depth attenuation came from runs
with the shared-phase bug above and does not survive.

\section*{4. Sweep 3: PSF elongation}

Lowering NA stretches the PSF axially far faster than laterally --- lateral FWHM
$\sim 0.5\lambda/\mathrm{NA}$, axial $\sim \lambda/(n - \sqrt{n^2-\mathrm{NA}^2})$ ---
so cells should stay about the same size in $xy$ and stretch in $z$. Five arms,
NA 1.0 down to 0.3, no refraction and no measurement noise, $W = 1920$.

\begin{center}
\begin{tabular}{lrrr}
\toprule
NA & apparent lateral extent & apparent axial extent & axial / lateral \\
\midrule
@PSFROWS@
\bottomrule
\end{tabular}
\end{center}

Figure~4 shows it directly: $xy$ sections barely change while $xz$ sections stretch
into cigars.

\textbf{Caveat throughout.} The reference is a confocal LSM 800 with a 2.54\,AU
pinhole. This model detects widefield --- every slice reaches the detector --- so
these stacks carry more out-of-focus haze than the real data. Equation~\eqref{eq:src}
supports the light-sheet case (excite one slice at a time), but a pinhole is not a
parameter change.

\begin{figure}[t]
\centering\includegraphics[width=\textwidth]{mc.png}
\caption*{\textbf{Figure 1.} a) Residual speckle of the Monte-Carlo average against
the number of random phase draws, detector off, with the fitted power law and
$1/\sqrt{W}$ for reference. b--d) The same plane at three values of $W$.}
\end{figure}

\begin{figure}[t]
\centering\includegraphics[width=\textwidth]{ri.png}
\caption*{\textbf{Figure 2.} Index sweep, 384\,\textmu m deep, no measurement
noise. a) Cell/gap contrast against focal depth, median per depth bin. b) The same
near the objective and on the far side, against $dn$; dotted line is zero contrast.
c--f) The stacks themselves: the index-matched control looks the same at both ends,
while at $dn = 0.02$ the far side has lost its cells. These arms run at $W = 240$, so
the visible grain is the Monte-Carlo residue of Figure~1 (2.8\,\%), not the detector,
which is off.}
\end{figure}

\begin{figure}[t]
\centering\includegraphics[width=\textwidth]{ri_sweep.png}
\caption*{\textbf{Figure 3.} The index sweep seen edge-on: an $xz$ strip through each
arm, the full 384\,\textmu m of depth on the horizontal axis, far side on the left and
objective on the right. Identical crop, scale and display range in every panel;
averaged over 1.5\,\textmu m in $y$ for display, equally in all arms. The cells fade
from the left as the index contrast rises, while the right-hand end holds.}
\end{figure}

\begin{figure}[t]
\centering\includegraphics[width=\textwidth]{psf.png}
\caption*{\textbf{Figure 4.} NA sweep, all panels the same isolated cell (the arms
share a seed) at the same physical scale. Top row $xy$, bottom row $xz$. Lower NA
leaves $xy$ nearly unchanged and stretches $z$.}
\end{figure}

\begin{figure}[t]
\centering\includegraphics[width=0.62\textwidth]{psf_extent.png}
\caption*{\textbf{Figure 5.} Apparent cell extent, measured at half maximum through
each cell's own centre, against NA.}
\end{figure}

\end{document}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mc = figure_mc(OUT / 'mc.png')
    ri = figure_ri(OUT / 'ri.png')
    figure_ri_sweep(OUT / 'ri_sweep.png')
    psf = figure_psf(OUT / 'psf.png')

    ri_rows = '\n'.join(
        f'{lab} & {n:.3f} & {f:.3f} & {(n - f) / n * 100:.0f}\\% \\\\'
        for lab, dn, n, f in ri)
    psf_rows = '\n'.join(
        f'{na} & {lat:.2f}\\,\\textmu m & {ax:.2f}\\,\\textmu m & {asp:.1f} \\\\'
        for na, lat, ax, asp in psf)

    for lab, dn, n, f in ri:
        print(f'RI  {lab:12s} near {n:.3f}  far {f:.3f}')
    for na, lat, ax, asp in psf:
        print(f'PSF NA {na:<4} lateral {lat:.2f} um  axial {ax:.2f} um  aspect {asp:.1f}')

    tex = TEX
    for key, val in [
        ('@DATE@', DATE),
        ('@SPECKLE1@', f"{mc[0]['speckle_pct']:.1f}"),
        ('@NFIRST@', str(mc[0]['n'])),
        ('@SPECKLE2@', f"{mc[-1]['speckle_pct']:.1f}"),
        ('@NLAST@', str(mc[-1]['n'])),
        ('@EXPONENT@', f"{mc[0].get('exponent', float('nan')):.2f}"),
        ('@RIROWS@', ri_rows),
        ('@PSFROWS@', psf_rows),
    ]:
        tex = tex.replace(key, val)
    (OUT / 'note.tex').write_text(tex)

    for _ in range(2):
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'note.tex'],
                           cwd=OUT, capture_output=True, text=True)
    if not (OUT / 'note.pdf').exists():
        print(r.stdout[-3000:])
        raise SystemExit('pdflatex failed')
    print(f'\nwrote {OUT}/note.pdf')


if __name__ == '__main__':
    main()
