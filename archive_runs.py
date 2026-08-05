"""Freeze runs into self-contained archive folders.

Each archive holds what is needed to read the result without the repo: the config as
run, the animations and figures, the ground-truth table, the measured analysis, and a
copy of the code that produced it. The TIFF volumes stay in log/ -- they are hundreds
of MB and the archive is meant to be readable, not a second copy of the dataset.

    python archive_runs.py                 # archive every log/ folder
    python archive_runs.py log/cells_dense
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

CODE = ['fluorescence_bpm.py', 'simulate.py', 'analyze_run.py', 'run_all.sh',
        'test_determinism.py', 'test_registration.py']
CODE_DIRS = ['fluo_bpm']
COPY = ['config.yaml', 'summary.md', 'summary.json', 'analysis.md', 'analysis.json',
        'gt_cells.csv', 'psf_table.csv', 'fluo_with_meas_noise.gif', 'fluo_without_meas_noise.gif',
        'fluo_projections_with_meas_noise.png', 'fluo_projections_without_meas_noise.png',
        'fluo_psf_grid.png']

# archive number -> log folder, in the order the experiments were done
ORDER = ['beads_psf', 'beads_psf_aberrated', 'cells_dense',
         'cells_dense_aberrated', 'cells_spheroid', 'cells_real_like',
         'cells_real_like_tall']


def git_rev():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def archive_one(log_dir: Path, archive_dir: Path):
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        src = log_dir / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)

    code = archive_dir / 'code'
    code.mkdir(exist_ok=True)
    for f in CODE:
        if Path(f).exists():
            shutil.copy2(f, code / f)
    for d in CODE_DIRS:
        if Path(d).exists():
            shutil.copytree(d, code / d, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__'))

    summary = json.loads((log_dir / 'summary.json').read_text())
    analysis_path = log_dir / 'analysis.json'
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    (archive_dir / 'README.md').write_text(readme(summary, analysis, log_dir))
    return summary, analysis


def readme(s, a, log_dir):
    voxel = s['voxel_um']
    lines = [
        f"# {s['name']}", '',
        s['description'].strip(), '',
        '## As run', '',
        f"- git rev `{s['git_rev']}`, seed `{s['seed']}`",
        f"- {s['n_objects']} objects, fill fraction {s['fill_fraction']*100:.2f}%",
        f"- delivered stack `{s['stack_shape_zyx']}` (z, y, x) at voxel "
        f"{voxel[0]} x {voxel[1]} x {voxel[2]} um",
        f"- lateral resolution {s['sampling']['lateral_resolution_um']:.2f} um, "
        f"axial {s['sampling']['axial_resolution_um']:.2f} um",
        f"- peak {s['photons']['peak_photons']:.0f} photons; "
        + (f"bright-voxel SNR {s['photons']['snr_bright']:.1f}"
           if s['photons'].get('noise_applied', True) else 'no detector noise'),
        f"- ran in {s['timing_s']['total']:.0f} s",
    ]
    if 'psf' in s:
        lines.append(f"- PSF over the bead grid: lateral FWHM "
                     f"{s['psf']['fwhm_xy_um_mean']:.2f} +- {s['psf']['fwhm_xy_um_std']:.2f} um, "
                     f"axial {s['psf']['fwhm_z_um_mean']:.2f} +- {s['psf']['fwhm_z_um_std']:.2f} um")
    if a:
        rt = a.get('brightness_correlation_integrated')
        lines += ['', '## Measured against ground truth', '',
                  f"- detection efficiency, surface / deepest octant: "
                  f"{a['efficiency_surface_vs_deep']:.2f}",
                  f"- emission recovery correlation: "
                  + ('n/a (uniform brightness)' if rt is None else f'{rt:.3f}'),
                  f"- brightest voxel vs true centre: lateral median "
                  f"{a['offset_lateral_um_median']:.2f} um, axial median "
                  f"{a['offset_axial_um_median']:.1f} um "
                  f"(bounded below by the {a['voxel_z_um']:.0f} um voxel)",
                  f"- nearest-neighbour distance median "
                  f"{a['nearest_neighbour_um_median']:.1f} um, "
                  f"{a['crowded_fraction']*100:.1f}% closer than the axial resolution"]
        if a.get('noise'):
            n = a['noise']
            lines.append(f"- noise check: measured std {n['measured_std']:.1f} vs "
                         f"shot+read prediction {n['predicted_std']:.1f} "
                         f"(ratio {n['ratio']:.2f})")
    lines += ['', '## Files', '',
              '- `fluo.gif` — green z-sweep of the delivered stack',
              '- `fluo_projections.png` — xy and xz projections against the ground truth',
              '- `analysis.md` — detection efficiency, emission recovery, offsets',
              '- `gt_cells.csv` — per-cell ground truth (position, radius, brightness)',
              '- `config.yaml` — the configuration as run; `code/` — the code that ran it',
              '', f"Volumes (`fluo.tif`, "
              f"`fluo_gt.tif`, `fluo_labels.tif`) stay in `{log_dir}/`.", '',
              'Reproduce with:', '', '```bash',
              f"python simulate.py -c config/{s['name']}.yaml", '```', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folders', nargs='*', default=None)
    p.add_argument('--archive_dir', default='archive')
    args = p.parse_args()

    folders = [Path(f) for f in args.folders] if args.folders else \
        [Path('log') / n for n in ORDER if (Path('log') / n).exists()]

    root = Path(args.archive_dir)
    rows = []
    for folder in folders:
        if not (folder / 'summary.json').exists():
            print(f'skip {folder}: no summary.json')
            continue
        idx = ORDER.index(folder.name) + 3 if folder.name in ORDER else 90
        dest = root / f'{idx:02d}_{folder.name}'
        s, a = archive_one(folder, dest)
        rows.append((dest.name, s, a))
        print(f'archived {folder} -> {dest}')

    if rows:
        index = ['# Archive', '',
                 f'Runs of the dataset engine, git rev `{git_rev()}`. Each folder is '
                 'self-contained: config, animations, figures, ground-truth table, '
                 'measured analysis, and the code that produced it.', '',
                 '| archive | cells | voxel [um] | noise | eff surface/deep | '
                 'r(emission) | notes |', '|---|---|---|---|---|---|---|']
        for name, s, a in rows:
            v = s['voxel_um']
            noise = ('shot+read' if s['photons'].get('noise_applied', True) else 'none')
            rt = a.get('brightness_correlation_integrated')
            eff = a.get('efficiency_surface_vs_deep', float('nan'))
            index.append(f"| [{name}]({name}/) | {s['n_objects']} | "
                         f"{v[0]:g}x{v[1]:g}x{v[2]:g} | {noise} | {eff:.2f} | "
                         f"{'n/a' if rt is None else f'{rt:.3f}'} | "
                         f"{s['description'].strip().splitlines()[0][:70]} |")
        (root / 'README.md').write_text('\n'.join(index) + '\n')
        print(f'wrote {root}/README.md')


if __name__ == '__main__':
    main()
