"""The engine: read one YAML, simulate, write one log folder.

    python simulate.py -c config/cells_dense.yaml

Log folder contents (log/<config name>/):

    config.yaml           the resolved configuration, as run
    gt_cells.csv          per-cell ground truth: position, radius, brightness
    fluo_gt.tif             true fluorophore concentration, on the delivered voxel grid
    fluo_labels.tif         per-cell label map, binned (nearest-cell attribution)
    fluo_without_noise.tif  simulated stack, no detector noise, float32 photons
    fluo_with_noise.tif     simulated stack with shot + read noise, uint16 ADU
    fluo_fine.tif           the un-binned simulation grid (only if save_fine_volume)
    fluo_with_noise.gif     green z-sweep of the noisy stack
    fluo_without_noise.gif  green z-sweep of the noise-free stack
    fluo_projections.png    xy/xz projections of truth, stack, noisy stack
    fluo_psf_grid.png       PSF width vs (x, y, z)              [phantom.type=beads]
    psf_table.csv           per-bead PSF fits                   [phantom.type=beads]
    summary.md              parameters, timings, photon statistics, findings
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tifffile import imwrite
from tqdm import tqdm

from fluorescence_bpm import Config as BPMConfig, FluorescenceBPM, set_deterministic

from . import camera as cam
from . import phantom as ph
from . import render
from .config import FluoBPMConfig, PhantomType
from .optics import sampling_report, zernike_pupil
from .psf import fit_bead_psfs, write_psf_table


def _git_rev():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def run(config: FluoBPMConfig, out_dir: str = None) -> dict:
    t_start = time.time()

    name = Path(config.config_file).stem if config.config_file != 'none' else 'run'
    out = Path(out_dir or Path(config.engine.log_dir) / name)
    out.mkdir(parents=True, exist_ok=True)

    if config.engine.deterministic:
        if config.seed is None:
            raise ValueError('engine.deterministic requires an explicit seed')
        set_deterministic(config.seed)

    vol, opt, cm = config.volume, config.optics, config.camera
    report = sampling_report(vol.dx, vol.dz, opt.lbda, opt.nm, opt.na)
    print(f"\n=== {name} — {config.description} ===")
    print(f"simulation grid {vol.nx} x {vol.ny} x {vol.nz} at "
          f"{vol.dx} x {vol.dx} x {vol.dz} um  "
          f"= {vol.extent_um[0]:.0f} x {vol.extent_um[1]:.0f} x {vol.extent_um[2]:.0f} um^3")
    print(f"delivered voxel {vol.dx*cm.bin_xy} x {vol.dx*cm.bin_xy} x {vol.dz*cm.bin_z} um, "
          f"stack {vol.nx//cm.bin_xy} x {vol.ny//cm.bin_xy} x {vol.nz//cm.bin_z}")
    print(f"resolution: lateral {report['lateral_resolution_um']:.2f} um, "
          f"axial {report['axial_resolution_um']:.2f} um  "
          f"(pupil {report['pupil_cutoff_um^-1']:.2f} < Nyquist {report['nyquist_um^-1']:.2f} um^-1)")

    # ---- sample -----------------------------------------------------------
    t0 = time.time()
    fluo, dn, labels, cells = ph.build(config)
    t_phantom = time.time() - t0
    fill = float((labels > 0).mean())
    print(f"phantom: {len(cells)} objects, fill fraction {fill*100:.2f}%, "
          f"dn range [{dn.min():.4f}, {dn.max():.4f}]  ({t_phantom:.1f} s)")
    if len(cells) < config.phantom.n_cells and config.phantom.type != PhantomType.BEADS:
        print(f"\033[93m[phantom] placed {len(cells)} of {config.phantom.n_cells} "
              f"requested — packing limit reached\033[0m")

    # The physics core works in [x, y, z], and it injects the source of a plane and
    # *then* propagates one step, so its focal planes are the source grid shifted by
    # one dz. Padding one empty plane at the entrance absorbs that shift exactly:
    # after the flip below, image plane k is focused on phantom plane k, with no
    # half-voxel offset, no dropped plane and no wrap-around.
    device = config.engine.device
    pad = np.zeros((1,) + fluo.shape[1:], dtype=np.float32)
    fluo_t = torch.tensor(np.moveaxis(np.concatenate([pad, fluo], axis=0), 0, -1), device=device)
    dn_t = torch.tensor(np.moveaxis(np.concatenate([dn[:1], dn], axis=0), 0, -1), device=device)
    nz_sim = vol.nz + 1

    # ---- optics -----------------------------------------------------------
    bpm_config = BPMConfig(
        nm=opt.nm, na=opt.na, dx=vol.dx, lbda=opt.lbda, dz=vol.dz,
        z_min=0.0, z_max=nz_sim * vol.dz,
        device=device, directions=[list(d) for d in opt.directions],
        stochastic=config.emission.stochastic, sparsity=config.emission.sparsity,
        seed=config.seed, deterministic=False,   # already applied above
    )
    model = FluorescenceBPM(bpm_config, dn=dn_t, fluo=fluo_t)
    model.pupil_aberration = zernike_pupil(model.mux, model.muy, opt.na, opt.lbda, opt.zernike)
    if model.pupil_aberration is not None:
        print(f"pupil aberration: "
              f"{ {k: v for k, v in opt.zernike.model_dump().items() if abs(v) > 0} } waves RMS")

    # ---- emission Monte Carlo --------------------------------------------
    t0 = time.time()
    n_dirs = len(opt.directions)
    I_total = torch.zeros_like(model.dn)
    for n in tqdm(range(config.emission.n_iterations), desc='emission'):
        phi = model.sample_phase()
        with torch.no_grad():
            for d in range(n_dirs):
                I_total += model(field_number=d, phi=phi)
    t_sim = time.time() - t0
    print(f"simulation: {config.emission.n_iterations} realizations x {n_dirs} direction(s) "
          f"in {t_sim:.1f} s ({config.emission.n_iterations/t_sim:.1f} it/s)")

    # The core returns the focal stack indexed from the exit face inward: its plane i
    # is focused on source plane Nz - i (verified with single emitters). Flip it, and
    # with the guard plane above image plane k lands on phantom plane k. Without this
    # flip every cell in gt_cells.csv would be mirrored in z against the stack.
    I_total = torch.flip(I_total, dims=[2])[:, :, :vol.nz]

    # ---- detector ---------------------------------------------------------
    binned = cam.bin_volume(I_total, cm.bin_xy, cm.bin_z)
    photons, scale = cam.to_photons(binned, cm)
    noisy_adu = cam.add_noise(photons, cm, generator=model.generator)
    stats = cam.snr_report(photons, cm)
    snr_txt = ('no detector noise' if not stats['noise_applied']
               else f"SNR {stats['snr_bright']:.1f}"
                    + ('' if cm.poisson else ' (read noise only, Poisson OFF)'))
    print(f"photons: peak {stats['peak_photons']:.0f}, bright-voxel mean "
          f"{stats['mean_bright_photons']:.0f}, {snr_txt}")

    # ground truth on the same delivered grid
    gt_binned = cam.bin_volume(torch.tensor(np.moveaxis(fluo, 0, -1), device=device),
                               cm.bin_xy, cm.bin_z) / (cm.bin_xy ** 2 * cm.bin_z)

    def to_zyx(t):
        return np.moveaxis(t.cpu().numpy(), -1, 0)

    clean_zyx = to_zyx(photons)
    noisy_zyx = np.moveaxis(cam.digitize(noisy_adu, cm), -1, 0)
    gt_zyx = to_zyx(gt_binned)

    # ---- write ------------------------------------------------------------
    (out / 'config.yaml').write_text(config.pretty())
    imwrite(out / 'fluo_without_noise.tif', clean_zyx.astype(np.float32))
    imwrite(out / 'fluo_with_noise.tif', noisy_zyx)
    if config.engine.save_gt_volume:
        imwrite(out / 'fluo_gt.tif', gt_zyx.astype(np.float32))
    if config.engine.save_label_volume:
        labels_binned = labels[::cm.bin_z, ::cm.bin_xy, ::cm.bin_xy]
        imwrite(out / 'fluo_labels.tif', labels_binned)
    if config.engine.save_fine_volume:
        imwrite(out / 'fluo_fine.tif', to_zyx(I_total).astype(np.float32))

    voxel = (vol.dx * cm.bin_xy, vol.dx * cm.bin_xy, vol.dz * cm.bin_z)
    with open(out / 'gt_cells.csv', 'w') as f:
        f.write(ph.GT_COLUMNS + ',x_vox,y_vox,z_vox\n')
        for c in cells:
            f.write(f'{c.label},{c.x:.4f},{c.y:.4f},{c.z:.4f},{c.r:.4f},{c.brightness:.5f},'
                    f'{c.x/voxel[0]:.4f},{c.y/voxel[1]:.4f},{c.z/voxel[2]:.4f}\n')

    # ---- PSF report -------------------------------------------------------
    psf = None
    if config.engine.psf_report:
        psf = fit_bead_psfs(config, cells, I_total)
        write_psf_table(psf, out / 'psf_table.csv')
        render.save_psf_figure(psf, out / 'fluo_psf_grid.png')
        print(f"PSF: lateral FWHM {np.nanmean(psf['fwhm_xy_um']):.2f} +- "
              f"{np.nanstd(psf['fwhm_xy_um']):.2f} um, axial "
              f"{np.nanmean(psf['fwhm_z_um']):.2f} +- {np.nanstd(psf['fwhm_z_um']):.2f} um "
              f"over {len(psf['z_um'])} beads")

    # ---- figures ----------------------------------------------------------
    if config.engine.save_gif:
        n_frames = render.save_gif(
            noisy_zyx.astype(np.float32), out / 'fluo_with_noise.gif',
            gamma=config.engine.gif_gamma, duration=config.engine.gif_duration_ms,
            max_frames=config.engine.gif_max_frames,
            labels=[f'z = {i*voxel[2]:.0f} um' for i in range(noisy_zyx.shape[0])],
        )
        render.save_gif(
            clean_zyx, out / 'fluo_without_noise.gif',
            gamma=config.engine.gif_gamma, duration=config.engine.gif_duration_ms,
            max_frames=config.engine.gif_max_frames,
            labels=[f'z = {i*voxel[2]:.0f} um' for i in range(clean_zyx.shape[0])],
        )
        render.save_projection_figure(clean_zyx, noisy_zyx, gt_zyx,
                                      out / 'fluo_projections.png', voxel[0], voxel[2])
        print(f"gif: {n_frames} frames")

    t_total = time.time() - t_start
    summary = {
        'name': name,
        'description': config.description,
        'git_rev': _git_rev(),
        'seed': config.seed,
        'n_objects': len(cells),
        'fill_fraction': fill,
        'sampling': report,
        'photons': stats,
        'photon_scale': scale,
        'voxel_um': voxel,
        'stack_shape_zyx': list(noisy_zyx.shape),
        'timing_s': {'phantom': t_phantom, 'simulation': t_sim, 'total': t_total},
    }
    if psf is not None:
        summary['psf'] = {
            'fwhm_xy_um_mean': float(np.nanmean(psf['fwhm_xy_um'])),
            'fwhm_xy_um_std': float(np.nanstd(psf['fwhm_xy_um'])),
            'fwhm_z_um_mean': float(np.nanmean(psf['fwhm_z_um'])),
            'fwhm_z_um_std': float(np.nanstd(psf['fwhm_z_um'])),
        }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    _write_summary_md(out / 'summary.md', config, summary)
    print(f"wrote {out}  ({t_total:.1f} s total)")
    return summary


def _write_summary_md(path, config: FluoBPMConfig, s: dict):
    vol, cm = config.volume, config.camera
    lines = [
        f"# {s['name']}", '',
        config.description, '',
        f"- git rev `{s['git_rev']}`, seed `{s['seed']}`",
        f"- simulation grid {vol.nx} x {vol.ny} x {vol.nz} at {vol.dx} / {vol.dz} um",
        f"- delivered stack {s['stack_shape_zyx']} at voxel "
        f"{s['voxel_um'][0]} x {s['voxel_um'][1]} x {s['voxel_um'][2]} um",
        f"- {s['n_objects']} objects, fill fraction {s['fill_fraction']*100:.2f}%",
        f"- NA {config.optics.na}, lambda {config.optics.lbda} um, n {config.optics.nm}; "
        f"lateral resolution {s['sampling']['lateral_resolution_um']:.2f} um, "
        f"axial {s['sampling']['axial_resolution_um']:.2f} um",
        f"- {config.emission.n_iterations} Monte Carlo realizations"
        + (f", STORM sparsity {config.emission.sparsity}" if config.emission.stochastic else ''),
        f"- medium `{config.medium.type}`, cell dn {config.phantom.dn_cell}",
        f"- Poisson noise {'on' if cm.poisson else 'off'}, read noise "
        f"{cm.read_noise_e} e, peak {s['photons']['peak_photons']:.0f} photons, "
        + (f"bright-voxel SNR {s['photons']['snr_bright']:.1f}"
           if s['photons']['noise_applied'] else 'no detector noise'),
        f"- timing: phantom {s['timing_s']['phantom']:.1f} s, simulation "
        f"{s['timing_s']['simulation']:.1f} s, total {s['timing_s']['total']:.1f} s",
    ]
    if 'psf' in s:
        lines.append(f"- PSF over the bead grid: lateral FWHM "
                     f"{s['psf']['fwhm_xy_um_mean']:.2f} +- {s['psf']['fwhm_xy_um_std']:.2f} um, "
                     f"axial {s['psf']['fwhm_z_um_mean']:.2f} +- {s['psf']['fwhm_z_um_std']:.2f} um")
    path.write_text('\n'.join(lines) + '\n')
