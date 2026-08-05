"""Run the Fluo-BPM engine on a YAML configuration.

    python simulate.py -c config/cells_dense.yaml
    python simulate.py -c config/cells_dense.yaml -o log/my_variant --device cuda:0
"""
import argparse
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before torch inits CUDA

from fluo_bpm.config import FluoBPMConfig
from fluo_bpm.engine import run

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', type=str, required=True)
    p.add_argument('-o', '--out_dir', type=str, default=None)
    p.add_argument('--device', type=str, default=None, help='override engine.device')
    p.add_argument('--n_iterations', type=int, default=None, help='override emission.n_iterations')
    p.add_argument('--seed', type=int, default=None, help='override seed')
    args = p.parse_args()

    config = FluoBPMConfig.from_yaml(args.config)
    if args.device is not None:
        config.engine.device = args.device
    if args.n_iterations is not None:
        config.emission.n_iterations = args.n_iterations
    if args.seed is not None:
        config.seed = args.seed

    run(config, out_dir=args.out_dir)
