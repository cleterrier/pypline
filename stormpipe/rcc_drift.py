# rcc_drift.py
"""
Redundant cross-correlation drift correction for STORM/SMLM localizations.

This module is a Python implementation inspired by SMAP's driftcorrectionXYZ,
finddriftfeature, finddriftfeatureZ, and applydriftcorrection MATLAB routines.

Expected localization table columns by default:
    frame, xnm, ynm, znm

Frames are assumed to be positive integer frame numbers. The returned drift table
contains one row per frame and can be used directly for plotting or QC.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.optimize import least_squares
from scipy.interpolate import interp1d, UnivariateSpline


@dataclass
class RCCDriftConfig:
    """Configuration for redundant cross-correlation drift correction."""

    # XY settings, corresponding to SMAP driftcorrectionXYZ settings
    correct_xy: bool = True
    xy_timepoints: int = 20
    xy_pixel_size_nm: float = 5.0
    xy_peak_window_pix: int = 7
    max_drift_nm: float = 1000.0
    max_reconstruction_size_pix: int = 4096

    # Optional Gaussian smoothing of rendered XY block images.
    # SMAP's histrenderc.c path used by this drift correction is effectively
    # nearest-pixel accumulation, not Gaussian rendering. Leave this as None
    # for the closest SMAP-like behavior; set e.g. 10-30 nm only if sparse
    # data need a smoother cross-correlation peak.
    xy_render_sigma_nm: Optional[float] = None

    # Z settings
    correct_z: bool = True
    z_timepoints: int = 20
    z_bin_width_nm: float = 5.0
    z_peak_window_pix: int = 9
    z_range_nm: Tuple[float, float] = (-400.0, 400.0)
    z_slice_width_nm: float = 200.0

    # Interpolation/smoothing settings
    # "spline" uses a cubic smoothing spline; "linear" uses linear interpolation.
    smooth_mode: str = "spline"
    # If None, an automatic moderate smoothing is used. Increase for smoother traces.
    # For exact interpolation, use smooth_mode="linear" or set spline_smoothing=0.
    spline_smoothing: Optional[float] = None

    # SMAP option "reference is last frame"; SMAP subtracts the penultimate
    # timepoint value, not the final one, so we mimic that behavior.
    reference_last: bool = False

    # Safety/diagnostics
    min_locs_per_timepoint: int = 500
    require_min_locs: bool = True
    robust_huber_c: float = 1.345
    robust_iterations: int = 20


def correct_drift_xyz(
    locs: pd.DataFrame,
    config: RCCDriftConfig = RCCDriftConfig(),
    *,
    frame_col: str = "frame",
    x_col: str = "xnm",
    y_col: str = "ynm",
    z_col: str = "znm",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Correct XY and optionally Z drift in a localization table.

    Frame-origin behavior
    ---------------------
    The RCC frame axis follows the actual frame numbers in `locs`.
    Therefore both 0-based and 1-based localization tables are handled correctly.
    """

    _validate_columns(locs, [frame_col, x_col, y_col])

    frames = locs[frame_col].to_numpy(dtype=float)
    finite_frames = frames[np.isfinite(frames)]
    if finite_frames.size == 0:
        raise ValueError(f"No finite frame values found in column {frame_col!r}")

    first_frame = int(np.nanmin(finite_frames))
    max_frame = int(np.nanmax(finite_frames))

    if first_frame < 0:
        raise ValueError(
            f"RCC drift correction expects non-negative frame numbers, got first_frame={first_frame}"
        )

    # Use the actual frame axis present in the localization table.
    # This works for 0-based, 1-based, and cropped frame ranges.
    drift_frames = np.arange(first_frame, max_frame + 1, dtype=int)

    drift_df = pd.DataFrame({"frame": drift_frames})
    drift_df["dx_nm"] = 0.0
    drift_df["dy_nm"] = 0.0
    drift_df["dz_nm"] = 0.0

    corrected = locs.copy()
    info: Dict[str, Any] = {"config": asdict(config)}

    if config.correct_xy:
        xy = estimate_xy_drift(
            corrected,
            config,
            frame_col=frame_col,
            x_col=x_col,
            y_col=y_col,
            max_frame=max_frame,
        )

        drift_df["dx_nm"] = xy.drift_x_nm[drift_frames]
        drift_df["dy_nm"] = xy.drift_y_nm[drift_frames]

        corrected[x_col] = corrected[x_col].to_numpy(dtype=float) - np.interp(
            frames,
            drift_df["frame"].to_numpy(dtype=float),
            drift_df["dx_nm"].to_numpy(dtype=float),
        )
        corrected[y_col] = corrected[y_col].to_numpy(dtype=float) - np.interp(
            frames,
            drift_df["frame"].to_numpy(dtype=float),
            drift_df["dy_nm"].to_numpy(dtype=float),
        )

        info["xy"] = xy.info

    if config.correct_z and z_col in corrected.columns:
        z_values = corrected[z_col].to_numpy()
        if np.isfinite(z_values).any():
            z = estimate_z_drift(
                corrected,
                config,
                frame_col=frame_col,
                x_col=x_col,
                z_col=z_col,
                max_frame=max_frame,
            )

            drift_df["dz_nm"] = z.drift_z_nm[drift_frames]

            corrected[z_col] = corrected[z_col].to_numpy(dtype=float) - np.interp(
                frames,
                drift_df["frame"].to_numpy(dtype=float),
                drift_df["dz_nm"].to_numpy(dtype=float),
            )

            info["z"] = z.info

    return corrected, drift_df, info


@dataclass
class XYDriftResult:
    drift_x_nm: np.ndarray  # length max_frame + 1; index by frame
    drift_y_nm: np.ndarray
    info: Dict[str, Any]


@dataclass
class ZDriftResult:
    drift_z_nm: np.ndarray  # length max_frame + 1; index by frame
    info: Dict[str, Any]


def estimate_xy_drift(
    locs: pd.DataFrame,
    config: RCCDriftConfig,
    *,
    frame_col: str = "frame",
    x_col: str = "xnm",
    y_col: str = "ynm",
    max_frame: Optional[int] = None,
) -> XYDriftResult:
    """Estimate per-frame XY drift using redundant cross-correlation."""

    _validate_columns(locs, [frame_col, x_col, y_col])
    data = locs[[frame_col, x_col, y_col]].dropna()
    frames = data[frame_col].to_numpy(dtype=int)
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    first_frame = int(frames.min())
    last_frame = int(frames.max())
    if max_frame is None:
        max_frame = last_frame

    if config.require_min_locs:
        locs_per_tp = len(data) / max(config.xy_timepoints, 1)
        if locs_per_tp < config.min_locs_per_timepoint:
            raise ValueError(
                f"Too few localizations per XY time window ({locs_per_tp:.1f}); "
                "increase signal, lower xy_timepoints, disable require_min_locs, or restrict less."
            )

    bin_frames, frame_edges = _make_frame_edges(first_frame, last_frame, config.xy_timepoints)
    movie_fft, render_info = _make_xy_fft_movie(
        frames,
        x,
        y,
        frame_edges,
        pixel_size_nm=config.xy_pixel_size_nm,
        max_size_pix=config.max_reconstruction_size_pix,
        render_sigma_nm=config.xy_render_sigma_nm,
    )

    ddx_pix, ddy_pix, peak_errors = _pairwise_xy_displacements(
        movie_fft,
        pixel_size_nm=config.xy_pixel_size_nm,
        peak_window_pix=config.xy_peak_window_pix,
        max_drift_nm=config.max_drift_nm,
    )

    ddx_nm = ddx_pix * config.xy_pixel_size_nm
    ddy_nm = ddy_pix * config.xy_pixel_size_nm

    dx_tp_nm, dx_solver = _solve_redundant_displacements(ddx_nm, config)
    dy_tp_nm, dy_solver = _solve_redundant_displacements(ddy_nm, config)

    ddx_plot = ddx_nm + dx_tp_nm[None, :]
    ddy_plot = ddy_nm + dy_tp_nm[None, :]
    sdx = _timepoint_scatter(ddx_plot)
    sdy = _timepoint_scatter(ddy_plot)

    centers = _timepoint_centers(first_frame, bin_frames, len(dx_tp_nm))
    frame_axis = np.arange(first_frame, int(max_frame) + 1, dtype=float)

    drift_x = _interpolate_trace(centers, dx_tp_nm, frame_axis, sdx, config)
    drift_y = _interpolate_trace(centers, dy_tp_nm, frame_axis, sdy, config)

    if config.reference_last and len(dx_tp_nm) > 1:
        ref_index = -2 if len(dx_tp_nm) >= 2 else -1
        drift_x = drift_x - dx_tp_nm[ref_index]
        drift_y = drift_y - dy_tp_nm[ref_index]
        dx_tp_nm = dx_tp_nm - dx_tp_nm[ref_index]
        dy_tp_nm = dy_tp_nm - dy_tp_nm[ref_index]

    drift_x_indexed = np.zeros(int(max_frame) + 1, dtype=float)
    drift_y_indexed = np.zeros(int(max_frame) + 1, dtype=float)

    frame_indices = frame_axis.astype(int)
    drift_x_indexed[frame_indices] = drift_x
    drift_y_indexed[frame_indices] = drift_y

    info = {
        "bin_frames": bin_frames,
        "frame_edges": frame_edges,
        "timepoint_centers": centers,
        "timepoint_dx_nm": dx_tp_nm,
        "timepoint_dy_nm": dy_tp_nm,
        "timepoint_sdx_nm": sdx,
        "timepoint_sdy_nm": sdy,
        "pairwise_dx_nm": ddx_nm,
        "pairwise_dy_nm": ddy_nm,
        "ddx_plot_nm": ddx_plot,
        "ddy_plot_nm": ddy_plot,
        "peak_errors": peak_errors,
        "solver_x": dx_solver,
        "solver_y": dy_solver,
        "render": render_info,
    }
    return XYDriftResult(drift_x_indexed, drift_y_indexed, info)


def estimate_z_drift(
    locs: pd.DataFrame,
    config: RCCDriftConfig,
    *,
    frame_col: str = "frame",
    x_col: str = "xnm",
    z_col: str = "znm",
    max_frame: Optional[int] = None,
) -> ZDriftResult:
    """Estimate per-frame Z drift using x-binned z cross-correlations."""

    _validate_columns(locs, [frame_col, x_col, z_col])
    data = locs[[frame_col, x_col, z_col]].dropna()
    frames = data[frame_col].to_numpy(dtype=int)
    x = data[x_col].to_numpy(dtype=float)
    z = data[z_col].to_numpy(dtype=float)

    # SMAP uses zrange for histograms but not as a strict localization filter.
    first_frame = int(frames.min())
    last_frame = int(frames.max())
    if max_frame is None:
        max_frame = last_frame

    bin_frames, frame_edges = _make_frame_edges(first_frame, last_frame, config.z_timepoints)
    z_edges = np.arange(
        config.z_range_nm[0],
        config.z_range_nm[1] + config.z_bin_width_nm,
        config.z_bin_width_nm,
        dtype=float,
    )
    x_edges = np.arange(np.nanmin(x), np.nanmax(x) + config.z_slice_width_nm, config.z_slice_width_nm)
    if len(x_edges) < 2:
        x_edges = np.array([np.nanmin(x), np.nanmin(x) + config.z_slice_width_nm], dtype=float)

    ddz_nm = _pairwise_z_displacements(
        frames,
        x,
        z,
        frame_edges,
        x_edges,
        z_edges,
        peak_window_pix=config.z_peak_window_pix,
    )

    dz_tp_nm, z_solver = _solve_redundant_displacements(ddz_nm, config)
    ddz_plot = ddz_nm + dz_tp_nm[None, :]
    sdz = _timepoint_scatter(ddz_plot)

    centers = _timepoint_centers(first_frame, bin_frames, len(dz_tp_nm))
    frame_axis = np.arange(first_frame, int(max_frame) + 1, dtype=float)
    drift_z = _interpolate_trace(centers, dz_tp_nm, frame_axis, sdz, config)

    if config.reference_last and len(dz_tp_nm) > 1:
        ref_index = -2 if len(dz_tp_nm) >= 2 else -1
        drift_z = drift_z - dz_tp_nm[ref_index]
        dz_tp_nm = dz_tp_nm - dz_tp_nm[ref_index]

    drift_z_indexed = np.zeros(int(max_frame) + 1, dtype=float)

    frame_indices = frame_axis.astype(int)
    drift_z_indexed[frame_indices] = drift_z

    info = {
        "bin_frames": bin_frames,
        "frame_edges": frame_edges,
        "timepoint_centers": centers,
        "timepoint_dz_nm": dz_tp_nm,
        "timepoint_sdz_nm": sdz,
        "pairwise_dz_nm": ddz_nm,
        "ddz_plot_nm": ddz_plot,
        "solver_z": z_solver,
        "x_edges_nm": x_edges,
        "z_edges_nm": z_edges,
    }
    return ZDriftResult(drift_z_indexed, info)


def _make_frame_edges(first_frame: int, last_frame: int, requested_timepoints: int) -> Tuple[int, np.ndarray]:
    """SMAP-like temporal binning, with the final edge inclusive in Python."""
    num_frames = last_frame - first_frame + 1
    bin_frames = int(2 * np.ceil(num_frames / max(requested_timepoints, 1) / 2 + 1))
    bin_frames = max(bin_frames, 1)

    # Use last_frame + 1 so the final frame is included with '< next_edge'.
    edges = list(range(first_frame, last_frame + 1, bin_frames))
    if edges[-1] != last_frame + 1:
        edges.append(last_frame + 1)
    edges = np.array(edges, dtype=int)

    # Remove accidental duplicate edges.
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([first_frame, last_frame + 1], dtype=int)
    return bin_frames, edges


def _timepoint_centers(first_frame: int, bin_frames: int, n_timepoints: int) -> np.ndarray:
    """SMAP-like timepoint center positions."""
    return (np.arange(n_timepoints, dtype=float) * bin_frames) + (bin_frames / 2.0) + first_frame


def _make_xy_fft_movie(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    frame_edges: np.ndarray,
    *,
    pixel_size_nm: float,
    max_size_pix: int,
    render_sigma_nm: Optional[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render each time block into an image and return its FFT."""

    x_work = np.asarray(x, dtype=float).copy()
    y_work = np.asarray(y, dtype=float).copy()
    mx = [float(np.nanmin(x_work)), float(np.nanmax(x_work))]
    my = [float(np.nanmin(y_work)), float(np.nanmax(y_work))]
    nx = max(1, int(np.round((mx[1] - mx[0]) / pixel_size_nm)))
    ny = max(1, int(np.round((my[1] - my[0]) / pixel_size_nm)))

    folded = False
    if max(nx, ny) > max_size_pix:
        # SMAP folds the reconstruction if it would be too large. We use the
        # intended behavior for both axes.
        folded = True
        max_nm = max_size_pix * pixel_size_nm
        x_work = np.mod(x_work - np.nanmin(x_work), max_nm)
        y_work = np.mod(y_work - np.nanmin(y_work), max_nm)
        mx = [float(np.nanmin(x_work)), float(np.nanmax(x_work))]
        my = [float(np.nanmin(y_work)), float(np.nanmax(y_work))]
        nx = max(1, int(np.round((mx[1] - mx[0]) / pixel_size_nm)))
        ny = max(1, int(np.round((my[1] - my[0]) / pixel_size_nm)))

    nfft = int(2 ** np.ceil(np.log2(max(max(nx, ny), 256))))
    if nfft > 2500:
        nfft = int(np.round(max(nx, ny) / 2.0) * 2)
        nfft = max(nfft, max(nx, ny))

    x_edges = mx[0] + np.arange(nx + 1, dtype=float) * pixel_size_nm
    y_edges = my[0] + np.arange(ny + 1, dtype=float) * pixel_size_nm

    n_blocks = len(frame_edges) - 1
    movie_fft = np.zeros((n_blocks, nfft, nfft), dtype=np.complex64)

    sigma_pix = None if render_sigma_nm is None else float(render_sigma_nm) / float(pixel_size_nm)

    for i in range(n_blocks):
        mask = (frames >= frame_edges[i]) & (frames < frame_edges[i + 1])
        img = _render_xy_histogram(
            x_work[mask],
            y_work[mask],
            x_min_nm=mx[0],
            y_min_nm=my[0],
            pixel_size_nm=pixel_size_nm,
            shape=(ny, nx),
        )
        if sigma_pix is not None and sigma_pix > 0:
            img = gaussian_filter(img, sigma=sigma_pix, mode="constant")
        movie_fft[i] = fft2(img, s=(nfft, nfft))

    info = {
        "x_range_nm": tuple(mx),
        "y_range_nm": tuple(my),
        "image_shape_yx": (ny, nx),
        "nfft": nfft,
        "folded": folded,
        "render_sigma_nm": render_sigma_nm,
    }
    return movie_fft, info


def _render_xy_histogram(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_min_nm: float,
    y_min_nm: float,
    pixel_size_nm: float,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Render localizations with SMAP histrenderc-like nearest-pixel accumulation.

    In histrender.m, normalized pixel coordinates are passed to histrenderc as
    single(xpix - 1), and the C code does `xr = xpix[k] + 0.5` before assigning
    to an integer. For positive coordinates this is effectively nearest-pixel
    binning with a half-pixel offset, rather than np.histogram's left-edge binning.
    """
    img = np.zeros(shape, dtype=np.float32)
    if len(x) == 0:
        return img

    # Emulate C's cast-to-long truncation toward zero.
    x_norm = (np.asarray(x, dtype=float) - float(x_min_nm)) / float(pixel_size_nm)
    y_norm = (np.asarray(y, dtype=float) - float(y_min_nm)) / float(pixel_size_nm)
    xi = np.trunc(x_norm - 0.5).astype(int)
    yi = np.trunc(y_norm - 0.5).astype(int)

    valid = (xi >= 0) & (xi < shape[1]) & (yi >= 0) & (yi < shape[0])
    np.add.at(img, (yi[valid], xi[valid]), 1.0)
    return img


def _pairwise_xy_displacements(
    movie_fft: np.ndarray,
    *,
    pixel_size_nm: float,
    peak_window_pix: int,
    max_drift_nm: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Calculate pairwise XY displacement matrices in reconstruction pixels."""
    n_blocks, nfft, _ = movie_fft.shape
    ddx = np.zeros((n_blocks, n_blocks), dtype=float)
    ddy = np.zeros((n_blocks, n_blocks), dtype=float)
    errx = np.zeros_like(ddx)
    erry = np.zeros_like(ddy)

    half_window = int(np.ceil((peak_window_pix - 1) / 2))
    max_drift_pix = float(max_drift_nm) / float(pixel_size_nm)
    center = nfft // 2

    for i in range(n_blocks - 1):
        for j in range(i + 1, n_blocks):
            cc = movie_fft[i] * np.conj(movie_fft[j])
            ccf = fftshift(ifft2(cc).real)
            peak = _find_xy_peak_gaussian(
                ccf,
                half_window=half_window,
                center=center,
                max_drift_pix=max_drift_pix,
            )
            ddx[i, j] = peak.shift_x_pix
            ddy[i, j] = peak.shift_y_pix
            ddx[j, i] = -peak.shift_x_pix
            ddy[j, i] = -peak.shift_y_pix
            errx[i, j] = errx[j, i] = peak.err_x_pix
            erry[i, j] = erry[j, i] = peak.err_y_pix

    return ddx, ddy, {"errx_pix": errx, "erry_pix": erry}


@dataclass
class _Peak2D:
    shift_x_pix: float
    shift_y_pix: float
    err_x_pix: float
    err_y_pix: float


def _find_xy_peak_gaussian(
    img: np.ndarray,
    *,
    half_window: int,
    center: int,
    max_drift_pix: float,
) -> _Peak2D:
    """Find cross-correlation maximum with subpixel 2D Gaussian fitting."""
    nrows, ncols = img.shape
    r0 = max(0, int(np.floor(center - max_drift_pix)))
    r1 = min(nrows, int(np.ceil(center + max_drift_pix)) + 1)
    c0 = max(0, int(np.floor(center - max_drift_pix)))
    c1 = min(ncols, int(np.ceil(center + max_drift_pix)) + 1)

    search = img[r0:r1, c0:c1]
    if search.size == 0:
        return _Peak2D(0.0, 0.0, np.inf, np.inf)

    # SMAP lightly filters before finding the maximum.
    search_smooth = uniform_filter(search, size=5, mode="nearest")
    rel_r, rel_c = np.unravel_index(np.nanargmax(search_smooth), search_smooth.shape)
    peak_r = r0 + rel_r
    peak_c = c0 + rel_c

    rr0 = max(0, peak_r - half_window)
    rr1 = min(nrows, peak_r + half_window + 1)
    cc0 = max(0, peak_c - half_window)
    cc1 = min(ncols, peak_c + half_window + 1)
    small = np.asarray(img[rr0:rr1, cc0:cc1], dtype=float)

    if min(small.shape) < 3 or not np.isfinite(small).all():
        return _Peak2D(float(peak_c - center), float(peak_r - center), np.inf, np.inf)

    try:
        fit_x, fit_y, err_x, err_y = _fit_elliptical_gaussian_2d(small)
        global_c = cc0 + fit_x
        global_r = rr0 + fit_y
        return _Peak2D(float(global_c - center), float(global_r - center), err_x, err_y)
    except Exception:
        # Fallback: quadratic interpolation around the brightest pixel.
        sub_x, sub_y = _quadratic_subpixel_peak(small)
        return _Peak2D(float(cc0 + sub_x - center), float(rr0 + sub_y - center), np.inf, np.inf)


def _fit_elliptical_gaussian_2d(image: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit A*exp(-0.5*[V11 dx^2 + 2 V12 dxdy + V22 dy^2]) + bg."""
    nrows, ncols = image.shape
    yy, xx = np.mgrid[0:nrows, 0:ncols]

    bg0 = float(np.nanmin(image))
    amp0 = float(np.nanmax(image) - bg0)
    amp0 = max(amp0, 1e-6)

    weights = image - bg0
    weights = np.clip(weights, 0, None)
    if np.sum(weights) <= 0:
        x0 = (ncols - 1) / 2.0
        y0 = (nrows - 1) / 2.0
        sx = sy = max(1.0, min(nrows, ncols) / 4.0)
    else:
        x0 = float(np.sum(xx * weights) / np.sum(weights))
        y0 = float(np.sum(yy * weights) / np.sum(weights))
        sx = float(np.sqrt(np.sum(((xx - x0) ** 2) * weights) / np.sum(weights)))
        sy = float(np.sqrt(np.sum(((yy - y0) ** 2) * weights) / np.sum(weights)))
        sx = max(sx, 1.0)
        sy = max(sy, 1.0)

    # Parameterize with log precision terms to keep widths positive and avoid
    # invalid covariance matrices. rho is constrained via tanh.
    p0 = np.array([x0, y0, amp0, bg0, np.log(1.0 / sx**2), np.log(1.0 / sy**2), 0.0])

    def residuals(p: np.ndarray) -> np.ndarray:
        xcen, ycen, amp, bg, log_vx, log_vy, rho_raw = p
        vx = np.exp(log_vx)
        vy = np.exp(log_vy)
        rho = 0.95 * np.tanh(rho_raw)
        vxy = rho * np.sqrt(vx * vy)
        expo = -0.5 * (
            vx * (xx - xcen) ** 2
            + 2.0 * vxy * (xx - xcen) * (yy - ycen)
            + vy * (yy - ycen) ** 2
        )
        model = amp * np.exp(expo) + bg
        return (model - image).ravel()

    lower = np.array([-image.shape[1], -image.shape[0], -np.inf, -np.inf, -12, -12, -4])
    upper = np.array([2 * image.shape[1], 2 * image.shape[0], np.inf, np.inf, 12, 12, 4])
    res = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=200,
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
    )

    xfit, yfit = float(res.x[0]), float(res.x[1])
    err_x = err_y = np.inf

    # Approximate parameter uncertainties from the Jacobian.
    if res.jac is not None and res.jac.size:
        dof = max(1, image.size - len(res.x))
        rss = float(np.sum(res.fun**2))
        try:
            cov = np.linalg.pinv(res.jac.T @ res.jac) * (rss / dof)
            err_x = float(2 * np.sqrt(max(cov[0, 0], 0.0)))
            err_y = float(2 * np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            pass

    return xfit, yfit, err_x, err_y


def _quadratic_subpixel_peak(image: np.ndarray) -> Tuple[float, float]:
    """2D fallback: independent parabolic interpolation in x and y."""
    r, c = np.unravel_index(np.nanargmax(image), image.shape)

    def one_dim_delta(vals: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(vals) - 1:
            return 0.0
        denom = vals[idx - 1] - 2 * vals[idx] + vals[idx + 1]
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (vals[idx - 1] - vals[idx + 1]) / denom

    dx = one_dim_delta(image[r, :], c)
    dy = one_dim_delta(image[:, c], r)
    return float(c + dx), float(r + dy)


def _pairwise_z_displacements(
    frames: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    frame_edges: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    *,
    peak_window_pix: int,
) -> np.ndarray:
    """SMAP-like pairwise z displacements from x-sliced z histograms."""
    n_blocks = len(frame_edges) - 1
    ddz = np.zeros((n_blocks, n_blocks), dtype=float)

    block_indices = [
        np.flatnonzero((frames >= frame_edges[i]) & (frames < frame_edges[i + 1]))
        for i in range(n_blocks)
    ]

    half_window = int(np.ceil((peak_window_pix - 1) / 2))
    for i in range(n_blocks - 1):
        idx_i = block_indices[i]
        for j in range(i + 1, n_blocks):
            idx_j = block_indices[j]
            shift = _find_displacement_z(
                x[idx_i],
                z[idx_i],
                x[idx_j],
                z[idx_j],
                x_edges=x_edges,
                z_edges=z_edges,
                half_window=half_window,
            )
            ddz[i, j] = shift
            ddz[j, i] = -shift
    return ddz


def _find_displacement_z(
    xr: np.ndarray,
    zr: np.ndarray,
    xt: np.ndarray,
    zt: np.ndarray,
    *,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    half_window: Optional[int],
) -> float:
    """Port of finddisplacementZ: sum x-sliced z cross-correlations."""
    n_z_bins = len(z_edges) - 1
    if n_z_bins < 3:
        return 0.0

    ccc = np.zeros(2 * n_z_bins - 1, dtype=float)

    for lo, hi in zip(x_edges[:-1], x_edges[1:]):
        zr_slice = zr[(xr >= lo) & (xr < hi)]
        zt_slice = zt[(xt >= lo) & (xt < hi)]
        hr, _ = np.histogram(zr_slice, bins=z_edges)
        ht, _ = np.histogram(zt_slice, bins=z_edges)
        hr = hr.astype(float) - np.mean(hr)
        ht = ht.astype(float) - np.mean(ht)
        ccc += np.convolve(hr, ht[::-1], mode="full")

    if not np.isfinite(ccc).all() or np.all(ccc == ccc[0]):
        return 0.0

    peak = int(np.argmax(ccc))
    if half_window is None:
        mc = ccc[peak]
        below = np.flatnonzero(ccc[peak:] < mc / 2.0)
        dh = max(3, int(np.round(below[0] / 2.0))) if len(below) else 3
    else:
        dh = max(1, int(half_window))

    bin_width = float(z_edges[1] - z_edges[0])
    lags = np.arange(-(n_z_bins - 1), n_z_bins, dtype=float) * bin_width

    lo = max(0, peak - dh)
    hi = min(len(ccc), peak + dh + 1)
    idx = np.arange(lo, hi)
    idx = idx[lags[idx] != 0]  # SMAP removes the zero lag point before fitting.

    if len(idx) < 3:
        return float(lags[peak])

    coeff = np.polyfit(lags[idx], ccc[idx], deg=2)
    a, b = coeff[0], coeff[1]
    if abs(a) < 1e-12:
        return float(lags[peak])
    return float(-b / (2.0 * a))


def _solve_redundant_displacements(
    dd: np.ndarray,
    config: RCCDriftConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Robustly solve dd[i, j] ~= d[i] - d[j], with d[0] fixed to 0.

    This replaces MATLAB nlinfit(..., 'Robust','on') with iteratively reweighted
    Huber least squares.
    """
    dd = np.asarray(dd, dtype=float)
    n = dd.shape[0]
    if n == 0:
        return np.array([], dtype=float), {"residuals": np.array([])}
    if n == 1:
        return np.array([0.0], dtype=float), {"residuals": np.array([])}

    rows = []
    b = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if not np.isfinite(dd[i, j]):
                continue
            row = np.zeros(n - 1, dtype=float)
            if i > 0:
                row[i - 1] += 1.0
            if j > 0:
                row[j - 1] -= 1.0
            rows.append(row)
            b.append(dd[i, j])

    if not rows:
        return np.zeros(n, dtype=float), {"residuals": np.array([])}

    A = np.vstack(rows)
    bvec = np.asarray(b, dtype=float)
    weights = np.ones_like(bvec)

    x = np.linalg.lstsq(A, bvec, rcond=None)[0]
    for _ in range(max(1, config.robust_iterations)):
        residuals = bvec - A @ x
        scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        scale = max(scale, 1e-9)
        threshold = config.robust_huber_c * scale
        weights = np.ones_like(residuals)
        large = np.abs(residuals) > threshold
        weights[large] = threshold / np.abs(residuals[large])

        Aw = A * np.sqrt(weights[:, None])
        bw = bvec * np.sqrt(weights)
        x_new = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        if np.linalg.norm(x_new - x) <= 1e-9 * (1 + np.linalg.norm(x)):
            x = x_new
            break
        x = x_new

    d = np.concatenate([[0.0], x])
    residuals = bvec - A @ x
    return d, {"residuals": residuals, "weights": weights}


def _timepoint_scatter(dd_plot: np.ndarray) -> np.ndarray:
    """Scatter of redundant estimates for each timepoint, used as spline weights."""
    if dd_plot.shape[1] <= 1:
        return np.ones(dd_plot.shape[0], dtype=float)
    scatter = np.nanstd(dd_plot, axis=1, ddof=0)
    finite = scatter[np.isfinite(scatter) & (scatter > 0)]
    floor = float(np.nanmedian(finite) / 2.0) if finite.size else 1.0
    floor = max(floor, 1e-6)
    scatter = np.where(np.isfinite(scatter) & (scatter > floor), scatter, floor)
    return scatter


def _interpolate_trace(
    centers: np.ndarray,
    values: np.ndarray,
    frames: np.ndarray,
    scatter: np.ndarray,
    config: RCCDriftConfig,
) -> np.ndarray:
    """Interpolate/smooth timepoint drift estimates onto every frame."""
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)
    frames = np.asarray(frames, dtype=float)

    good = np.isfinite(centers) & np.isfinite(values)
    if np.count_nonzero(good) == 0:
        return np.zeros_like(frames, dtype=float)
    if np.count_nonzero(good) == 1:
        return np.full_like(frames, values[good][0], dtype=float)

    x = centers[good]
    y = values[good]
    s = np.asarray(scatter, dtype=float)[good]
    s = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s[np.isfinite(s) & (s > 0)]))
    s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
    weights = 1.0 / s

    mode = config.smooth_mode.lower()
    if mode in {"linear", "interp", "interpolate"} or len(x) < 4:
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
        out = f(frames)
    elif mode in {"spline", "cubic_spline", "smoothing cubic spline"}:
        if config.spline_smoothing is None:
            # Moderate automatic smoothing. This is not numerically identical to
            # MATLAB csaps(p=[]), but gives a stable RCC drift trace.
            smoothing = max(len(x) - np.sqrt(2 * len(x)), 0.0)
        else:
            smoothing = float(config.spline_smoothing)
        spline = UnivariateSpline(x, y, w=weights, k=min(3, len(x) - 1), s=smoothing)
        out = spline(frames)
        # Clamp outside the measured center range, matching interp1/csaps edge handling
        # used by SMAP after interpolation.
        out = np.asarray(out, dtype=float)
        out[frames <= x[0]] = y[0]
        out[frames >= x[-1]] = y[-1]
    else:
        raise ValueError(f"Unknown smooth_mode: {config.smooth_mode!r}")

    return np.asarray(out, dtype=float)


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required localization column(s): {missing}")


def summarize_drift(drift: pd.DataFrame) -> pd.Series:
    """Convenience QC summary for a drift table returned by correct_drift_xyz."""
    out = {}
    for col in ["dx_nm", "dy_nm", "dz_nm"]:
        if col in drift:
            values = drift[col].to_numpy(dtype=float)
            out[f"{col}_range"] = float(np.nanmax(values) - np.nanmin(values))
            out[f"{col}_start"] = float(values[0])
            out[f"{col}_end"] = float(values[-1])
    return pd.Series(out)
