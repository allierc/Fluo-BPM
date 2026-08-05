"""
Run the fluorescence BPM on the spherical-cell phantom.

  python make_cell_phantom.py          # writes data/cells_{fluorescence,dn}.tif
  python run_cells.py                  # writes output/cells/fluorescence_final.tif
  python make_gif.py --input output/cells/fluorescence_final.tif
"""
import argparse
import os

from fluorescence_bpm import run_simulation

NZ = 64
DZ = 1.0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n_iterations', type=int, default=200)
    p.add_argument('--na', type=float, default=0.8)
    p.add_argument('--device', type=str, default='cuda:1')
    p.add_argument('--output_path', type=str, default='./output/cells')
    p.add_argument('--tag', type=str, default='cells')
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--sparsity', type=float, default=0.01)
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    run_simulation(
        refractive_index_path=f'./data/{args.tag}_dn.tif',
        fluorescence_path=f'./data/{args.tag}_fluorescence.tif',
        output_path=args.output_path,
        dx=0.2,               # matches make_cell_phantom.py
        dz=DZ,
        lbda=0.532,
        z_min=0.0,
        z_max=NZ * DZ,        # Nz must equal the volume depth
        nm=1.33,
        na=args.na,
        device=args.device,
        stochastic=args.stochastic,
        sparsity=args.sparsity,
        n_iterations=args.n_iterations,
    )
