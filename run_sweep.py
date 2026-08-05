"""Run a parameterized sweep.

    python run_sweep.py -s config/sweeps/RI_sweep.yaml
    python run_sweep.py -s config/sweeps/PSF_sweep.yaml --device cuda:0
    python run_sweep.py -s config/sweeps/MC_sweep.yaml --dry_run

Arms land in log/<sweep name>/<label>/. A sweep is one file, so reproducing it is one
command; the per-arm configuration is written into each arm's folder as config.yaml.
"""
import argparse
import json
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from fluo_bpm.engine import run
from fluo_bpm.sweep import Sweep

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--spec', required=True)
    p.add_argument('--device', default=None, help='override engine.device')
    p.add_argument('--log_dir', default='./log')
    p.add_argument('--dry_run', action='store_true', help='print the arms and stop')
    args = p.parse_args()

    sweep = Sweep(args.spec)
    print(f'=== {sweep.name}: {sweep.parameter} over {sweep.values} ===')

    summaries = []
    for label, value, config in sweep.arms():
        if args.device:
            config.engine.device = args.device
        out = sweep.out_dir(label, args.log_dir)
        if args.dry_run:
            print(f'  {label:8s} {sweep.parameter} = {value}  -> {out}')
            continue
        summaries.append(run(config, out_dir=str(out)))

    if summaries:
        index = {'sweep': sweep.name, 'parameter': sweep.parameter,
                 'arms': [{'label': l, 'value': v} for (l, v) in
                          zip(sweep.labels, sweep.values)],
                 'speckle_pct': [s.get('monte_carlo_speckle_pct') for s in summaries]}
        path = sweep.out_dir('', args.log_dir).parent / f'{sweep.name}.json'
        path.write_text(json.dumps(index, indent=2))
        print(f'wrote {path}')
