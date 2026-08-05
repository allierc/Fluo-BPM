"""Does a cell in gt_cells.csv sit where the image says it does?

Two things are easy to get wrong and impossible to see in a rendered stack:

1. Axial order. The physics core returns the focal stack indexed from the exit face
   inward -- its plane i is focused on source plane Nz - i -- so an unflipped stack
   is mirrored in z against the phantom. A mirrored ground truth still looks like a
   plausible volume of cells, so nothing downstream would complain.

2. Lateral order. `np.moveaxis(volume_zyx, 0, -1)` gives [y, x, z], not [x, y, z]:
   the core's first axis is the TIFF's y. With nx == ny this is invisible, and it
   round-trips correctly through the engine, but a test that assumes [x, y, z] will
   report a phantom failure (this is exactly what happened while writing the engine).

Run:  python test_registration.py
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from fluorescence_bpm import Config, FluorescenceBPM

GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'

NX = NY = 128
NZ = 64
DX, DZ = 0.5, 1.0
LBDA, NA, NM = 0.532, 0.5, 1.33


def simulate_single_emitter(k, y, x, device, n_iter=30):
    """One emitter in phantom plane k at TIFF position (y, x); returns the engine's
    image volume in [y, x, z] (the core's own axis order)."""
    fluo = np.zeros((NZ, NY, NX), dtype=np.float32)
    fluo[k, y, x] = 1.0
    padded = np.concatenate([np.zeros((1, NY, NX), dtype=np.float32), fluo], axis=0)
    dn = np.zeros_like(padded)

    config = Config(nm=NM, na=NA, dx=DX, lbda=LBDA, dz=DZ,
                    z_min=0.0, z_max=padded.shape[0] * DZ, device=device, seed=1)
    model = FluorescenceBPM(
        config,
        dn=torch.tensor(np.moveaxis(dn, 0, -1), device=device),
        fluo=torch.tensor(np.moveaxis(padded, 0, -1), device=device),
    )
    I = torch.zeros_like(model.dn)
    with torch.no_grad():
        for _ in range(n_iter):
            I += model(0, model.sample_phase())
    return torch.flip(I, dims=[2])[:, :, :NZ].cpu().numpy()   # engine convention


def main(device='cuda:1'):
    failures = []

    print('axial: phantom plane k -> brightest image plane')
    for k in (0, 10, 31, 50, 63):
        vol = simulate_single_emitter(k, 64, 30, device)
        prof = vol[62:67, 28:33, :].max(axis=(0, 1))
        got = int(prof.argmax())
        ok = got == k
        print(f'  k={k:3d} -> {got:3d}  {"ok" if ok else "MISMATCH"}')
        if not ok:
            failures.append(f'axial k={k} -> {got}')

    print('lateral: emitter at (y=90, x=30) -> image argmax')
    vol = simulate_single_emitter(31, 90, 30, device)
    iy, ix, iz = np.unravel_index(vol.argmax(), vol.shape)
    ok = (iy, ix, iz) == (90, 30, 31)
    print(f'  (y, x, z) = ({iy}, {ix}, {iz})  {"ok" if ok else "MISMATCH"}')
    if not ok:
        failures.append(f'lateral -> ({iy}, {ix}, {iz})')

    print()
    if failures:
        print(f'{RED}FAIL{RESET}  ' + '; '.join(failures))
    else:
        print(f'{GREEN}PASS{RESET}  image plane k is focused on phantom plane k, '
              f'lateral indices preserved')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'cuda:1')
