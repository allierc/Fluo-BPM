"""Build the sample: what is in the volume and what its refractive index looks like.

Everything is returned in [z, y, x] order, float32, on the CPU as numpy, so it can
be written straight to TIFF as ground truth.
"""
from typing import List, Tuple

import numpy as np

from .config import BrightnessDist, FluoBPMConfig, MediumType, PhantomType


class Cell:
    """One sphere. Positions and radii in um, measured from the volume corner."""
    __slots__ = ('x', 'y', 'z', 'r', 'brightness', 'label')

    def __init__(self, x, y, z, r, brightness, label):
        self.x, self.y, self.z, self.r = x, y, z, r
        self.brightness, self.label = brightness, label

    def as_row(self):
        return (self.label, self.x, self.y, self.z, self.r, self.brightness)


GT_COLUMNS = 'label,x_um,y_um,z_um,r_um,brightness'


def plane_depths(vol) -> np.ndarray:
    """Depth of each source plane [um].

    Planes sit at dz, 2*dz, ... nz*dz rather than at half-steps. The reason is the
    detection geometry of the physics core: the forward loop injects the source of
    plane s and then propagates one step, so after the loop the field sits one step
    past the last source. Back-propagating it gives focal planes at exactly
    dz, 2*dz, ... nz*dz, and this choice makes source plane s and image plane s the
    same physical depth with no half-voxel offset and no dropped plane.
    """
    return (np.arange(vol.nz) + 1.0) * vol.dz


def plane_of_depth(vol, z_um: float) -> int:
    """Inverse of plane_depths: index of the plane nearest a given depth."""
    return int(np.clip(round(z_um / vol.dz) - 1, 0, vol.nz - 1))


def _draw_brightness(cfg, rng, n):
    b = cfg.brightness
    if b.dist == BrightnessDist.LOGNORMAL:
        v = rng.lognormal(mean=0.0, sigma=b.sigma, size=n)
    elif b.dist == BrightnessDist.UNIFORM:
        v = rng.uniform(b.low, b.high, size=n)
    else:
        v = np.full(n, b.low)
    return np.clip(v, b.clip[0], b.clip[1])


def _grid_key(x, y, z, cell_size):
    return (int(x // cell_size), int(y // cell_size), int(z // cell_size))


def place_cells(config: FluoBPMConfig) -> List[Cell]:
    """Rejection-sample non-overlapping spheres.

    A uniform hash grid keeps this linear in the number of cells: at a few thousand
    cells an all-pairs test is ~10^7 distance evaluations per accepted cell.
    """
    ph, vol = config.phantom, config.volume
    rng = np.random.default_rng(config.seed if config.seed is not None else None)
    Lx, Ly, Lz = vol.extent_um

    r_min, r_max = ph.radius_um
    cell_size = 2 * r_max + ph.min_gap_um
    grid = {}

    def neighbours(x, y, z):
        i0, j0, k0 = _grid_key(x, y, z, cell_size)
        for i in (i0 - 1, i0, i0 + 1):
            for j in (j0 - 1, j0, j0 + 1):
                for k in (k0 - 1, k0, k0 + 1):
                    for c in grid.get((i, j, k), ()):
                        yield c

    if ph.type == PhantomType.SPHEROID:
        cx, cy, cz = ph.spheroid_center_um or (Lx / 2, Ly / 2, Lz / 2)
        R = ph.spheroid_radius_um
    cells: List[Cell] = []
    brightness = _draw_brightness(ph, rng, ph.n_cells)

    max_tries = 400 * ph.n_cells
    tries = 0
    while len(cells) < ph.n_cells and tries < max_tries:
        tries += 1
        r = rng.uniform(r_min, r_max)
        m = ph.margin_um + r
        if ph.type == PhantomType.SPHEROID:
            # uniform inside a ball, then reject what pokes outside the volume
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            rad = (R - r) * rng.uniform(0, 1) ** (1 / 3)
            x, y, z = cx + u[0] * rad, cy + u[1] * rad, cz + u[2] * rad
            if not (m <= x <= Lx - m and m <= y <= Ly - m and m <= z <= Lz - m):
                continue
        else:
            x = rng.uniform(m, Lx - m)
            y = rng.uniform(m, Ly - m)
            z = rng.uniform(m, Lz - m)

        ok = True
        for c in neighbours(x, y, z):
            if (x - c.x) ** 2 + (y - c.y) ** 2 + (z - c.z) ** 2 < (r + c.r + ph.min_gap_um) ** 2:
                ok = False
                break
        if not ok:
            continue

        cell = Cell(x, y, z, r, float(brightness[len(cells)]), len(cells) + 1)
        cells.append(cell)
        grid.setdefault(_grid_key(x, y, z, cell_size), []).append(cell)

    return cells


def place_beads(config: FluoBPMConfig) -> List[Cell]:
    """Sub-resolution emitters on a regular (x, y, z) grid: a PSF probe."""
    ph, vol = config.phantom, config.volume
    Lx, Ly, Lz = vol.extent_um
    gx, gy, gz = ph.bead_grid
    # inset by 10% of the extent so no bead sits on the FFT wrap boundary
    # Snap every bead to a voxel centre. A sub-resolution emitter rasterized off-grid
    # loses most of its amplitude to the soft edge (a 0.15 um bead half a plane off
    # centre comes out ~5x dimmer), which would show up as a spurious depth
    # dependence of the "PSF brightness" rather than of the PSF.
    depths = plane_depths(vol)
    xs = (np.round(np.linspace(0.1 * Lx, 0.9 * Lx, gx) / vol.dx - 0.5) + 0.5) * vol.dx
    ys = (np.round(np.linspace(0.1 * Ly, 0.9 * Ly, gy) / vol.dx - 0.5) + 0.5) * vol.dx
    zs = depths[np.round(np.linspace(0, vol.nz - 1, gz)).astype(int)]
    cells, label = [], 1
    for z in zs:
        for y in ys:
            for x in xs:
                cells.append(Cell(float(x), float(y), float(z), ph.bead_radius_um,
                                  ph.bead_brightness, label))
                label += 1
    return cells


def rasterize(config: FluoBPMConfig, cells: List[Cell]):
    """Cells -> (fluorescence, dn, labels) volumes, all [z, y, x].

    Each sphere is written with a tanh-softened edge so the phantom is not aliased;
    `labels` carries the integer cell id of the nearest-covering sphere, which is
    the ground truth an assessment needs to attribute recovered blobs to cells.
    """
    ph, vol = config.phantom, config.volume
    nz, ny, nx = vol.nz, vol.ny, vol.nx
    fluo = np.zeros((nz, ny, nx), dtype=np.float32)
    dn = np.zeros((nz, ny, nx), dtype=np.float32)
    labels = np.zeros((nz, ny, nx), dtype=np.uint16)

    x_ax = (np.arange(nx) + 0.5) * vol.dx
    y_ax = (np.arange(ny) + 0.5) * vol.dx
    z_ax = plane_depths(vol)
    edge = max(ph.edge_um, 1e-6)

    if ph.type == PhantomType.BEADS:
        # single-voxel deltas of equal amplitude: the PSF probe measures the optics,
        # so every emitter has to be the same emitter
        for c in cells:
            i = int(np.clip(round(c.x / vol.dx - 0.5), 0, nx - 1))
            j = int(np.clip(round(c.y / vol.dx - 0.5), 0, ny - 1))
            k = plane_of_depth(vol, c.z)
            fluo[k, j, i] = c.brightness
            labels[k, j, i] = c.label
            if ph.dn_cell:
                dn[k, j, i] = ph.dn_cell
        return fluo, dn, labels

    for c in cells:
        pad = c.r + 3 * edge
        i0, i1 = np.searchsorted(x_ax, [c.x - pad, c.x + pad])
        j0, j1 = np.searchsorted(y_ax, [c.y - pad, c.y + pad])
        k0, k1 = np.searchsorted(z_ax, [c.z - pad, c.z + pad])
        i1, j1, k1 = max(i1, i0 + 1), max(j1, j0 + 1), max(k1, k0 + 1)
        if i0 >= nx or j0 >= ny or k0 >= nz:
            continue

        X = x_ax[i0:i1].reshape(1, 1, -1) - c.x
        Y = y_ax[j0:j1].reshape(1, -1, 1) - c.y
        Z = z_ax[k0:k1].reshape(-1, 1, 1) - c.z
        dist = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

        mask = 0.5 * (1.0 - np.tanh((dist - c.r) / edge))
        if ph.hollow:
            inner = c.r - ph.shell_um
            if inner > 0:
                mask = mask * 0.5 * (1.0 + np.tanh((dist - inner) / edge))

        sub_f = fluo[k0:k1, j0:j1, i0:i1]
        sub_d = dn[k0:k1, j0:j1, i0:i1]
        sub_l = labels[k0:k1, j0:j1, i0:i1]
        contribution = (c.brightness * mask).astype(np.float32)
        np.maximum(sub_f, contribution, out=sub_f)
        # dn follows the solid sphere even when the fluorescence is a shell:
        # the whole cell body refracts light, only the label is in the cytoplasm
        solid = 0.5 * (1.0 - np.tanh((dist - c.r) / edge))
        np.maximum(sub_d, (ph.dn_cell * solid).astype(np.float32), out=sub_d)
        sub_l[solid > 0.5] = c.label

    return fluo, dn, labels


def medium_index(config: FluoBPMConfig) -> np.ndarray:
    """Refractive index landscape of the embedding medium, [z, y, x]."""
    med, vol = config.medium, config.volume
    nz, ny, nx = vol.nz, vol.ny, vol.nx

    if med.type == MediumType.NONE:
        return np.zeros((nz, ny, nx), dtype=np.float32)

    if med.type == MediumType.SMOOTH_RANDOM:
        # band-limited Gaussian field: white noise shaped by a Gaussian in k-space,
        # which gives an exactly prescribed correlation length in each axis
        rng = np.random.default_rng((config.seed or 0) + 991)
        noise = rng.standard_normal((nz, ny, nx)).astype(np.float32)
        kz = np.fft.fftfreq(nz, d=vol.dz).reshape(-1, 1, 1)
        ky = np.fft.fftfreq(ny, d=vol.dx).reshape(1, -1, 1)
        kx = np.fft.fftfreq(nx, d=vol.dx).reshape(1, 1, -1)
        lz, lxy = med.correlation_z_um, med.correlation_um
        envelope = np.exp(-((np.pi * kx * lxy) ** 2 + (np.pi * ky * lxy) ** 2
                           + (np.pi * kz * lz) ** 2))
        field = np.fft.ifftn(np.fft.fftn(noise) * envelope).real.astype(np.float32)
        field -= field.mean()
        rms = float(field.std()) or 1.0
        return (field * (med.dn_rms / rms)).astype(np.float32)

    if med.type == MediumType.SLAB:
        # a tilted index slab across the top of the volume: the classic
        # coverslip/mounting-medium mismatch, giving depth- and x-dependent aberration
        thickness = med.slab_thickness_um
        x_ax = (np.arange(nx) + 0.5) * vol.dx
        z_ax = (np.arange(nz) + 0.5) * vol.dz
        ramp = 1.0 + med.slab_tilt * (x_ax / (nx * vol.dx) - 0.5) * 2 * nx * vol.dx / thickness
        edge = np.clip(thickness * ramp, 1.0, None).reshape(1, 1, -1)
        zz = z_ax.reshape(-1, 1, 1)
        field = med.slab_dn * 0.5 * (1.0 - np.tanh((zz - edge) / 2.0))
        return np.broadcast_to(field, (nz, ny, nx)).astype(np.float32).copy()

    raise ValueError(f'unknown medium type {med.type}')


def build(config: FluoBPMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Cell]]:
    """(fluorescence, dn, labels, cells) for the configured phantom."""
    if config.phantom.type == PhantomType.BEADS:
        cells = place_beads(config)
    else:
        cells = place_cells(config)

    fluo, dn, labels = rasterize(config, cells)
    dn = dn + medium_index(config)
    return fluo, dn, labels, cells
