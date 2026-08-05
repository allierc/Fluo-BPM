"""From the simulation grid to a delivered image stack: binning, then noise.

Order matters. The simulation grid is finer than the delivered voxel (it has to be,
or the pupil does not fit in k-space), so photons are collected by binning first and
only then converted to counts. Adding Poisson noise before binning would model a
detector that samples at 0.5 um and averages, which suppresses the shot noise by
sqrt(bin_xy^2 * bin_z) and reports an SNR the real instrument does not have.
"""
import numpy as np
import torch

from .config import CameraConfig


def bin_volume(vol: torch.Tensor, bin_xy: int, bin_z: int) -> torch.Tensor:
    """[Nx, Ny, Nz] -> summed over bin_xy x bin_xy x bin_z blocks.

    Sum, not mean: binning collects photons.
    """
    nx, ny, nz = vol.shape
    v = vol.reshape(nx // bin_xy, bin_xy, ny // bin_xy, bin_xy, nz // bin_z, bin_z)
    return v.sum(dim=(1, 3, 5))


def to_photons(vol: torch.Tensor, camera: CameraConfig):
    """Scale so the brightest voxel carries camera.photons_peak photons, then add
    the background. Returns (photons, scale) so the same scale can be reused for a
    matched noise-free reference."""
    peak = float(vol.max())
    scale = camera.photons_peak / peak if peak > 0 else 1.0
    return vol * scale + camera.background_photons, scale


def add_noise(photons: torch.Tensor, camera: CameraConfig, generator=None):
    """Shot noise, then read noise, then offset and digitization.

    Returns a float tensor in ADU (not yet cast), so the caller can inspect the
    dynamic range before clipping to the bit depth.
    """
    if camera.poisson:
        counts = torch.poisson(photons.clamp(min=0), generator=generator)
    else:
        counts = photons.clamp(min=0)

    adu = counts * camera.gain
    if camera.read_noise_e > 0:
        noise = torch.randn(adu.shape, device=adu.device, dtype=adu.dtype,
                            generator=generator) * (camera.read_noise_e * camera.gain)
        adu = adu + noise
    return adu + camera.offset


def digitize(adu: torch.Tensor, camera: CameraConfig) -> np.ndarray:
    vmax = 2 ** camera.bit_depth - 1
    out = adu.clamp(0, vmax).round()
    dtype = np.uint16 if camera.bit_depth > 8 else np.uint8
    return out.cpu().numpy().astype(dtype)


def snr_report(clean_photons: torch.Tensor, camera: CameraConfig):
    """Photon statistics of the delivered stack, for the run summary."""
    bg = camera.background_photons
    signal = clean_photons - bg
    # "in a cell" = the top 2% of voxels, i.e. roughly the bright interiors
    thr = torch.quantile(signal.flatten().float()[::max(1, signal.numel() // 1_000_000)], 0.98)
    bright = signal[signal >= thr]
    mean_bright = float(bright.mean()) if bright.numel() else 0.0
    # only count the noise sources the configuration actually applies, otherwise a
    # noise-free run reports the SNR of a run that was never simulated
    shot = float(np.sqrt(mean_bright + bg)) if (camera.poisson and mean_bright > 0) else 0.0
    read = camera.read_noise_e
    total = float(np.sqrt(shot ** 2 + read ** 2))
    return {
        'peak_photons': float(clean_photons.max()),
        'background_photons': bg,
        'mean_bright_photons': mean_bright,
        'shot_noise_e': shot,
        'read_noise_e': read,
        'noise_applied': bool(camera.poisson or read > 0),
        'snr_bright': float(mean_bright / total) if total > 0 else float('inf'),
    }
