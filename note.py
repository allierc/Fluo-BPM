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

RI_SWEEP = [('dn0000', 0.0, 'dn = 0'), ('dn0025', 0.0025, 'dn = 0.0025'),
            ('dn0050', 0.005, 'dn = 0.005'), ('dn0100', 0.010, 'dn = 0.010'),
            ('dn0200', 0.020, 'dn = 0.020')]
PSF_SWEEP = [('na100', 1.0), ('na080', 0.8), ('na060', 0.6),
             ('na040', 0.4), ('na030', 0.3)]
OUT = Path('archive/note')
DATE = '2026-08-05'


def load(tag, root='log'):
    folder = Path(root) / tag
    summary = json.loads((folder / 'summary.json').read_text())
    config = yaml.safe_load((folder / 'config.yaml').read_text())
    stack = io.imread(folder / 'fluo.tif').astype(np.float32)
    gain = config['camera'].get('gain', 1.0) or 1.0
    pe = (stack - config['camera'].get('offset', 0.0)) / gain
    gt = io.imread(folder / 'fluo_gt.tif').astype(np.float32)
    cells = np.loadtxt(folder / 'gt_cells.csv', delimiter=',', skiprows=1)
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
    data = {tag: load(tag, root='log/RI_sweep') for tag, _, _ in RI_SWEEP}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), facecolor='black')
    for ax in axes:
        ax.set_facecolor('black'); ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_color('white')

    colours = ['#bbbbbb', '#8ad48a', '#5ad4c0', '#5a8ce0', '#e05a5a']
    near, far, dns = [], [], []
    for (tag, dn, label), colour in zip(RI_SWEEP, colours):
        d = data[tag]
        c = cell_contrast(d)
        depth = (np.arange(len(c)) + 1) * d['summary']['voxel_um'][2]
        axes[0].plot(depth, c, color=colour, lw=1.5, label=label)
        q = max(len(c) // 4, 1)
        near.append(np.nanmedian(c[-q:])); far.append(np.nanmedian(c[:q])); dns.append(dn)

    axes[0].axhline(0.0, color='white', ls=':', lw=1)
    axes[0].set_xlabel('focal depth [um]   (objective side at the right)', color='white')
    axes[0].set_ylabel('cell / gap contrast', color='white')
    axes[0].set_title('(a)  contrast against depth', color='white', fontsize=11, loc='left')
    axes[1].plot(dns, near, 'o-', color='#8ad48a', lw=1.6, label='near the objective')
    axes[1].plot(dns, far, 'o-', color='#e05a5a', lw=1.6, label='far side')
    axes[1].axhline(0.0, color='white', ls=':', lw=1)
    axes[1].set_xlabel('cell index contrast  dn', color='white')
    axes[1].set_ylabel('cell / gap contrast', color='white')
    axes[1].set_title('(b)  near and far contrast against dn', color='white',
                      fontsize=11, loc='left')
    for ax in axes:
        leg = ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=9)
        leg.get_frame().set_alpha(0.5)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor='black'); plt.close(fig)
    return [(lab, dn, n, f) for (_, dn, lab), n, f in zip(RI_SWEEP, near, far)]


def figure_psf(path):
    rows = []
    fig, axes = plt.subplots(2, len(PSF_SWEEP), figsize=(3.1 * len(PSF_SWEEP), 6.4),
                             facecolor='black')
    for col, (tag, na) in enumerate(PSF_SWEEP):
        d = load(tag, root='log/PSF_sweep')
        pe = d['pe']
        vx, vy, vz = d['summary']['voxel_um']
        nz, ny, nx = pe.shape
        lat, ax_ = cell_extent(d)
        rows.append((na, lat, ax_, ax_ / lat if lat else np.nan))

        xy = pe[nz // 2, ny // 2 - 64:ny // 2 + 64, nx // 2 - 64:nx // 2 + 64]
        # z along the horizontal axis, physically scaled: a tall thin xz panel is
        # unreadable in a row of five
        xz = pe[:, ny // 2, nx // 2 - 64:nx // 2 + 64].T
        for row, (img, aspect, tag_txt) in enumerate([
                (xy, 1.0, 'xy'), (xz, vx / vz, 'xz  (z horizontal)')]):
            a = axes[row, col]
            a.imshow(img, cmap='Greens_r', aspect=aspect, vmin=0,
                     vmax=np.percentile(img, 99.5))
            a.set_xticks([]); a.set_yticks([]); a.set_facecolor('black')
            a.set_title(f'({"abcdefghij"[row * len(PSF_SWEEP) + col]})  NA {na}, {tag_txt}',
                        color='white', fontsize=10, loc='left', pad=5)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor='black'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor='black')
    ax.set_facecolor('black'); ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('white')
    na = [r[0] for r in rows]
    ax.plot(na, [r[1] for r in rows], 'o-', color='#8ad48a', lw=1.6, label='lateral extent')
    ax.plot(na, [r[2] for r in rows], 'o-', color='#e05a5a', lw=1.6, label='axial extent')
    ax.set_xlabel('numerical aperture', color='white')
    ax.set_ylabel('apparent cell extent, FWHM [um]', color='white')
    ax.set_title('(a)  apparent cell size against NA', color='white', fontsize=11, loc='left')
    leg = ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=9)
    leg.get_frame().set_alpha(0.5)
    fig.tight_layout()
    fig.savefig(str(path).replace('.png', '_extent.png'), dpi=130, facecolor='black')
    plt.close(fig)
    return rows


def figure_mc(path):
    rows = json.loads(Path('log/mc_sweep/mc_sweep.json').read_text())
    n = np.array([r['n'] for r in rows], dtype=float)
    sp = np.array([r['speckle_pct'] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor='black')
    ax.set_facecolor('black'); ax.tick_params(colors='white')
    for s in ax.spines.values():
        s.set_color('white')
    ax.loglog(n, sp, 'o-', color='#e05a5a', lw=1.6, label='measured speckle')
    ax.loglog(n, sp[0] * np.sqrt(n[0] / n), ':', color='white', lw=1.3,
              label=r'$1/\sqrt{W}$ through the first point')
    ax.set_xlabel('random phase draws  $W$', color='white')
    ax.set_ylabel('residual speckle [% of local mean]', color='white')
    ax.set_title('(a)  speckle of the Monte-Carlo sum', color='white', fontsize=11, loc='left')
    leg = ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=9)
    leg.get_frame().set_alpha(0.5)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor='black'); plt.close(fig)
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
fluctuates by @SPECKLE1@\,\% of the local mean at $W = @NFIRST@$, falling as
$1/\sqrt{W}$ to @SPECKLE2@\,\% at $W = @NLAST@$ (Figure~1). Grain in a folder whose
name says no measurement noise is this. The sweeps below run at $W = 240$ and
$W = 1920$ respectively.

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
three quarters of the index-matched value (Figure~2). Two notes on measuring this.
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

Figure~3 shows it directly: $xy$ sections barely change while $xz$ sections stretch
into cigars.

\textbf{Caveat throughout.} The reference is a confocal LSM 800 with a 2.54\,AU
pinhole. This model detects widefield --- every slice reaches the detector --- so
these stacks carry more out-of-focus haze than the real data. Equation~\eqref{eq:src}
supports the light-sheet case (excite one slice at a time), but a pinhole is not a
parameter change.

\begin{figure}[t]
\centering\includegraphics[width=0.5\textwidth]{mc.png}
\caption*{\textbf{Figure 1.} Residual speckle of the Monte-Carlo average against the
number of random phase draws, detector off, against a $1/\sqrt{W}$ reference.}
\end{figure}

\begin{figure}[t]
\centering\includegraphics[width=\textwidth]{ri.png}
\caption*{\textbf{Figure 2.} Index sweep, 384\,\textmu m deep, no measurement noise.
(a) Cell/gap contrast against focal depth. (b) The same near the objective and on the
far side, against $dn$; the dotted line is zero contrast.}
\end{figure}

\begin{figure}[t]
\centering\includegraphics[width=\textwidth]{psf.png}
\caption*{\textbf{Figure 3.} NA sweep. Top row $xy$ sections, bottom row $xz$ sections
of the same volume, at the delivered voxel aspect. Lower NA leaves $xy$ nearly
unchanged and stretches $z$.}
\end{figure}

\begin{figure}[t]
\centering\includegraphics[width=0.5\textwidth]{psf_extent.png}
\caption*{\textbf{Figure 4.} Apparent cell extent, measured at half maximum through
each cell's own centre, against NA.}
\end{figure}

\end{document}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mc = figure_mc(OUT / 'mc.png')
    ri = figure_ri(OUT / 'ri.png')
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
