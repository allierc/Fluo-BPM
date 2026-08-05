"""Is a Fluo-BPM run byte-for-byte repeatable? Measure it, then confirm the fix.

The question
-----------
The Monte Carlo loop accumulates n_iterations random-phase realizations. Before
the seed work, both random draws in the model -- the emission phase screen and
the STORM activation mask -- came from the *global* torch RNG, unseeded. Two
runs of the same config therefore produced different volumes, so no output TIFF
could be re-generated and no figure made from one could be checked.

The fix
-------
`FluorescenceBPM` owns a `torch.Generator` seeded from `Config.seed`, and both
draws go through it. `Config.seed = None` restores the old behaviour (entropy).
`deterministic=True` additionally pins cuDNN/algorithm selection, which matters
because the accumulation `I_total += I` sums 1000 volumes and float addition is
not associative.

What this script does
---------------------
Runs the real simulation twice per arm on one GPU and compares the two runs
within each arm by SHA-256 of the output TIFF:

    arm 'entropy'  seed=None            -- expect the runs to differ
    arm 'seeded'   seed=42              -- expect the runs to be identical
    arm 'strict'   seed=42, determ.=True

Usage:
    python test_determinism.py --tag cells --n-iter 25
"""
import argparse
import hashlib
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before torch inits CUDA

import numpy as np
from tifffile import imwrite

GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'


def sha256_of(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def one_run(tag, n_iter, seed, deterministic, device, dx, dz, lbda, na, nz):
    from fluorescence_bpm import run_simulation
    out = './output/_dettest'
    os.makedirs(out, exist_ok=True)
    I = run_simulation(
        refractive_index_path=f'./data/{tag}_dn.tif',
        fluorescence_path=f'./data/{tag}_fluorescence.tif',
        output_path=out,
        dx=dx, dz=dz, lbda=lbda, nm=1.33, na=na,
        z_min=0.0, z_max=nz * dz,
        device=device,
        stochastic=True, sparsity=0.01,
        n_iterations=n_iter,
        seed=seed, deterministic=deterministic,
    )
    return I


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', type=str, default='cells')
    p.add_argument('--n-iter', type=int, default=25)
    p.add_argument('--runs', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda:1')
    p.add_argument('--dx', type=float, default=0.2)
    p.add_argument('--dz', type=float, default=1.0)
    p.add_argument('--lbda', type=float, default=0.532)
    p.add_argument('--na', type=float, default=0.8)
    p.add_argument('--nz', type=int, default=64)
    p.add_argument('--arms', type=str, default='entropy,seeded,strict')
    args = p.parse_args()

    arms = {
        'entropy': dict(seed=None, deterministic=False, expect_same=False),
        'seeded':  dict(seed=42, deterministic=False, expect_same=True),
        'strict':  dict(seed=42, deterministic=True, expect_same=True),
    }

    results = {}
    for name in args.arms.split(','):
        cfg = arms[name]
        print(f"\n=== arm '{name}'  seed={cfg['seed']} deterministic={cfg['deterministic']} ===")
        digests, vols = [], []
        for r in range(args.runs):
            I = one_run(args.tag, args.n_iter, cfg['seed'], cfg['deterministic'],
                        args.device, args.dx, args.dz, args.lbda, args.na, args.nz)
            digests.append(sha256_of(I))
            vols.append(I)
            print(f"  run {r}: sha256 {digests[-1][:16]}  sum {I.sum():.6e}")

        same = len(set(digests)) == 1
        max_delta = float(np.abs(vols[0] - vols[-1]).max())
        rel = max_delta / (float(np.abs(vols[0]).max()) + 1e-30)
        ok = (same == cfg['expect_same'])
        colour = GREEN if ok else RED
        print(f"  {colour}identical={same} (expected {cfg['expect_same']})  "
              f"max|delta|={max_delta:.3e}  rel={rel:.3e}{RESET}")
        results[name] = ok

    print('\n=== summary ===')
    for name, ok in results.items():
        print(f"  {GREEN if ok else RED}{'PASS' if ok else 'FAIL'}{RESET}  {name}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
