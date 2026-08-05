"""Detection-pupil aberration.

The BPM already gives a depth-dependent PSF (defocus is in the propagator) and,
whenever dn varies, a laterally varying one. This module adds the part that
belongs to the objective rather than to the sample: a Zernike phase on the
detection pupil.

Conventions: coordinates are the model's spatial frequencies mux, muy [um^-1] in
fftshifted order, normalized by the pupil radius NA/lbda, so rho = 1 at the rim.
Coefficients are in waves RMS (Noll-normalized polynomials).
"""
import numpy as np
import torch

from .config import ZernikeConfig


def _zernike_terms(rho, theta):
    """Noll-normalized polynomials used here, keyed by config field name."""
    return {
        'defocus':   np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0),                 # Z4
        'astig_0':   np.sqrt(6.0) * rho ** 2 * torch.cos(2 * theta),        # Z6
        'astig_45':  np.sqrt(6.0) * rho ** 2 * torch.sin(2 * theta),        # Z5
        'coma_x':    np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * torch.cos(theta),   # Z8
        'coma_y':    np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * torch.sin(theta),   # Z7
        'spherical': np.sqrt(5.0) * (6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0),           # Z11
    }


def zernike_pupil(mux, muy, na, lbda, zernike: ZernikeConfig):
    """Complex [Nx, Ny] pupil factor, or None when the aberration is zero.

    mux is [1, Nx] and muy is [Ny, 1] as the model stores them; the result
    broadcasts to the pupil array the model builds.
    """
    if not zernike.any_nonzero():
        return None

    radius = na / lbda
    mx = mux.reshape(1, -1)
    my = muy.reshape(-1, 1)
    rho = torch.sqrt(mx ** 2 + my ** 2) / radius
    theta = torch.atan2(my.expand_as(rho), mx.expand_as(rho))

    terms = _zernike_terms(rho, theta)
    phase = torch.zeros_like(rho)
    for name, coeff in zernike.model_dump().items():
        if abs(coeff) > 0:
            phase = phase + float(coeff) * terms[name]

    phase = 2 * np.pi * phase                      # waves -> radians
    phase = torch.where(rho <= 1.0, phase, torch.zeros_like(phase))  # outside the stop
    return torch.polar(torch.ones_like(phase), phase)


def sampling_report(dx, dz, lbda, nm, na):
    """What the grid can and cannot represent. Printed by the engine before a run."""
    nyquist = 1.0 / (2.0 * dx)
    cutoff = na / lbda
    K = nm / lbda
    mu2 = min(nyquist, cutoff) ** 2
    defocus_rad = 2 * np.pi * dz * (np.sqrt(max(K ** 2 - mu2, 0.0)) - K)
    return {
        'nyquist_um^-1': nyquist,
        'pupil_cutoff_um^-1': cutoff,
        'pupil_fits_in_kspace': bool(cutoff < nyquist),
        'defocus_phase_per_dz_rad': float(defocus_rad),
        'lateral_resolution_um': float(0.5 * lbda / na),
        'axial_resolution_um': float(lbda / (nm - np.sqrt(max(nm ** 2 - na ** 2, 0.0)))
                                    if na < nm else float('nan')),
        'max_na_for_dx': float(lbda / (2 * dx)),
    }
