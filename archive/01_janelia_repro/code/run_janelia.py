"""
Reproduce assets/fluo.gif from the recovered logo phantom (see make_janelia_phantom.py).

Two variants:

  --preset cryolite   the parameters in run_simulation.py (dx=2.412 um, NA=0.95).
                      NOTE: NA/lbda = 1.49 um^-1 while the Nyquist frequency is
                      1/(2*dx) = 0.21 um^-1, so the pupil mask passes the whole
                      sampled k-space and the defocus phase over one dz is 0.13 rad.
                      The model therefore has no optical sectioning here: every
                      output plane is the same projection of the volume.

  --preset optical    same phantom sampled at microscope resolution (dx=0.2 um), where
                      the pupil sits inside k-space and defocus is real, so the z-sweep
                      shows each depth coming into focus as in the published animation.
"""
import argparse
import os

from fluorescence_bpm import run_simulation

PRESETS = {
    'cryolite': dict(dx=2.412, dz=1.989, lbda=0.637, nm=1.33, na=0.95),
    'optical':  dict(dx=0.200, dz=1.989, lbda=0.637, nm=1.33, na=0.95),
}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=list(PRESETS), default='optical')
    p.add_argument('--nz', type=int, default=29)
    p.add_argument('--n_iterations', type=int, default=1000)
    p.add_argument('--sparsity', type=float, default=0.01)
    p.add_argument('--device', type=str, default='cuda:1')
    p.add_argument('--tag', type=str, default='janelia')
    p.add_argument('--output_path', type=str, default=None)
    args = p.parse_args()

    pr = PRESETS[args.preset]
    out = args.output_path or f'./output/{args.tag}_{args.preset}'
    os.makedirs(out, exist_ok=True)

    run_simulation(
        refractive_index_path=f'./data/{args.tag}_dn.tif',
        fluorescence_path=f'./data/{args.tag}_fluorescence.tif',
        output_path=out,
        z_min=0.0,
        z_max=args.nz * pr['dz'],
        device=args.device,
        stochastic=True,
        sparsity=args.sparsity,
        n_iterations=args.n_iterations,
        **pr,
    )
