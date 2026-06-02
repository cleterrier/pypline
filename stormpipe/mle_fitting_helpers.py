# mle_fitting_helpers.py

"""
GPU fitting helpers
-------------------
Low-level utilities used by the ratiometric STORM GPU fitting pipeline.

This module contains:
- spline PSF coefficient validation and orientation helpers
- low-level loclib MLE/DLL wrappers
- single-channel result formatting
- Stage 3 dTS builders for free-ratio and fixed-ratio global fits
- global dual-channel spline PSF model loading
- likelihood normalization helpers

Higher-level pipeline orchestration lives in:
    gpu_fitting_pipeline.py

Stage 3 run context, paired-ROI GPU-batch iterator setup, and CSV filename
helpers live in:
    pipeline_helpers.py
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy as np
import scipy.io

logger = logging.getLogger("stormpipe.gpu")

# =============================================================================
# loclib CUDA memory-layout boundary helpers
# =============================================================================
#
# The loclib CUDA ABI uses flat contiguous buffers whose logical order follows
# MATLAB-style array semantics:
#
#     data(x, y, fit[, channel])
#     coeff(x, y, z, basis64[, channel])
#
# Python therefore prepares C-contiguous NumPy arrays whose flattened memory
# matches that loclib logical order.
#
# Naming convention used here:
#   N = fit / ROI index
#   C = camera / channel index
#   X = image column / CUDA first spatial index
#   Y = image row    / CUDA second spatial index
#   Z = spline z index
#   B = spline basis index, usually 64
#
# Important distinction:
#   - NumPy images are naturally indexed image[y, x], so cut ROIs are (N, Y, X).
#   - loclib/CUDA expects flattened memory equivalent to MATLAB
#     data(x, y, fit[, channel]).
#
def rois_nyx_image_to_loclib_nxy(rois_nyx: np.ndarray) -> np.ndarray:
    """
    Convert image-order ROI batches to the single-channel loclib layout.

    Input
    -----
    rois_nyx
        NumPy/preprocessing image-order ROIs with shape (N, Y, X), where each
        ROI was cut from an image indexed as image[y, x].

    Output
    ------
    np.ndarray
        C-contiguous float32 array with shape (N, X, Y). Its flattened memory
        matches MATLAB's column-major data(x, y, fit), which is what the
        single-channel loclib CUDA binding copied directly to CUDA.
    """
    rois = np.asarray(rois_nyx, dtype=np.float32)

    if rois.ndim != 3:
        raise ValueError(
            f"rois_nyx must be a 3D array with shape (N,Y,X), got {rois.shape}"
        )

    return np.ascontiguousarray(rois.swapaxes(1, 2), dtype=np.float32)


def pack_dual_rois_nyx_image_to_loclib_cnxy(
    rois_R_nyx: np.ndarray,
    rois_T_nyx: np.ndarray,
) -> np.ndarray:
    """
    Pack paired R/T image-order ROIs for the multichannel loclib fitter.

    Input
    -----
    rois_R_nyx, rois_T_nyx
        Image-order ROI batches with shape (N, Y, X).

    Output
    ------
    np.ndarray
        C-contiguous float32 array with shape (C, N, X, Y), C=2. Its flattened
        memory matches MATLAB's column-major data(x, y, fit, channel), which is
        what the multichannel loclib CUDA binding copied directly to CUDA.
    """
    rois_R_nxy = rois_nyx_image_to_loclib_nxy(rois_R_nyx)
    rois_T_nxy = rois_nyx_image_to_loclib_nxy(rois_T_nyx)

    if rois_T_nxy.shape != rois_R_nxy.shape:
        raise ValueError(
            f"rois_T_nyx must match rois_R_nyx after conversion; "
            f"got {rois_T_nxy.shape} vs {rois_R_nxy.shape}"
        )

    return np.ascontiguousarray(
        np.stack([rois_R_nxy, rois_T_nxy], axis=0),
        dtype=np.float32,
    )


def coeff_xyzb_matlab_to_loclib_bzyx(coeff_xyzb: np.ndarray) -> np.ndarray:
    """
    Convert one MATLAB/SMAP spline coefficient block to loclib memory layout.

    Input logical MATLAB layout
    ---------------------------
    coeff(x, y, z, basis64), represented in Python as shape (X, Y, Z, B).

    Output NumPy layout
    -------------------
    Shape (B, Z, Y, X), C-contiguous. Its flattened memory is equivalent to the
    MATLAB column-major coeff(x, y, z, basis64) array expected by the CUDA spline
    code.
    """
    validate_spline_coeff_xyzw(coeff_xyzb)

    return np.ascontiguousarray(
        np.transpose(coeff_xyzb, (3, 2, 1, 0)),
        dtype=np.float32,
    )


def coeff_xyzb_matlab_to_single_channel_loclib_bzyx(
    coeff_xyzb: np.ndarray,
) -> np.ndarray:
    """
    Convert one Stage 1 spline coefficient block to single-channel loclib layout.

    Input logical layout
    --------------------
    coeff(x, y, z, basis64), represented in Python as shape (X, Y, Z, B).

    Output loclib layout
    --------------------
    Shape (B, Z, Y, X), C-contiguous.

    This is the same per-channel coefficient layout used by the global
    dual-channel spline model before channel stacking. Its flattened memory
    matches the loclib CUDA convention:

        coeff(x, y, z, basis64)
    """
    return coeff_xyzb_matlab_to_loclib_bzyx(coeff_xyzb)

# =============================================================================
# Single-channel spline PSF coefficient preparation
# =============================================================================
def validate_spline_coeff_xyzw(coeff_xyzw: np.ndarray, *, expect_basis: int = 64) -> tuple[int, int, int, int]:
    if coeff_xyzw.dtype != np.float32:
        raise ValueError(f"PSF coeff dtype must be float32, got {coeff_xyzw.dtype}")
    if coeff_xyzw.ndim != 4:
        raise ValueError(f"PSF coeff must be 4D, got {coeff_xyzw.shape}")
    X, Y, Z, B = map(int, coeff_xyzw.shape)
    if X != Y:
        raise ValueError(f"PSF coeff must be square in XY, got {X}x{Y}")
    if expect_basis is not None and B != expect_basis:
        raise ValueError(f"Expected last axis (basis) == {expect_basis}, got {B}")
    return X, Y, Z, B


def build_single_channel_splinesize_for_dll(
    coeff_loclib_single: np.ndarray,
) -> np.ndarray:
    """
    Build spline-size metadata for the single-channel loclib spline binding.

    The input coefficient array must already be packed in single-channel loclib
    layout:

        coeff_loclib_single.shape == (B, Z, Y, X)

    The returned metadata follows the DLL convention:

        [X, Y, Z, B]

    The coefficient tensor and this metadata must be passed to the DLL together
    as a matched pair.
    """
    coeff_loclib_single = np.asarray(coeff_loclib_single)

    if coeff_loclib_single.ndim != 4 or coeff_loclib_single.shape[0] != 64:
        raise ValueError(
            "Expected single-channel loclib coeff shape (B,Z,Y,X) with B=64, "
            f"got {coeff_loclib_single.shape}"
        )

    B, Z, Y, X = map(int, coeff_loclib_single.shape)
    return np.array([X, Y, Z, B], dtype=np.int32)


def zseed_center_to_absolute(
    coeff_loclib_single: np.ndarray,
    zstart_center_idx: float = 0.0,
) -> np.float32:
    """
    Convert a z-start offset from the spline center to an absolute spline index.

    The coefficient tensor is expected in single-channel loclib layout:

        coeff_loclib_single.shape == (B, Z, Y, X)
    """
    Z = int(coeff_loclib_single.shape[1])
    z_abs = np.float32(zstart_center_idx + Z / 2.0)
    eps = np.float32(1e-6)
    if z_abs < 0: z_abs = np.float32(0.0 + eps)
    if z_abs > Z - 1: z_abs = np.float32((Z - 1) - eps)
    return z_abs

# =============================================================================
# Low-level MLE / DLL wrappers
# =============================================================================
def mle_spline_single_channel(
    dll,
    rois_nyx: np.ndarray,  # image-order (N, Y, X) float32
    xpix_N: np.ndarray,
    ypix_N: np.ndarray,
    frame_N: np.ndarray,
    *,
    coeff_loclib_single: np.ndarray,
    splinesize_loclib_single: np.ndarray,
    iterations: int = 30,
    EMexcessNoise: int = 1,
    varstack: int | np.ndarray = 0,
    zseed_abs: np.float32 = None,
    flag6: int = 1,
    prealloc_P: np.ndarray | None = None,
    prealloc_CRLB: np.ndarray | None = None,
    prealloc_LogL: np.ndarray | None = None,
):
    """
    Low-level wrapper around the loclib single-channel spline MLE CUDA routine.

    Parameters
    ----------
    dll
        localizationlib(...) instance exposing `_mleFit`.
    rois_nyx
        Image-order ROI batch with shape (N, Y, X), float32 photons.
        The ROI batch is packed internally into loclib layout before the DLL call.
    xpix_N, ypix_N, frame_N
        Per-ROI seed metadata from the preprocessing stage.
    coeff_loclib_single
        Packed single-channel spline coefficients with shape (B, Z, Y, X).
    splinesize_loclib_single
        int32 metadata equivalent to MATLAB size(coeff): [X, Y, Z, B].
    iterations
        LM iteration cap.
    EMexcessNoise
        EM excess-noise scaling used by the loclib call and result formatter.
    varstack
        0 for EMCCD-like mode, or a per-pixel variance array with shape (Y, X, N).
    zseed_abs
        Initial absolute spline z index. If None, the spline center is used.
    prealloc_P, prealloc_CRLB, prealloc_LogL
        Optional preallocated output arrays.

    Returns
    -------
    P : (6, N) float32
        Fit parameters in loclib order [Y, X, Phot, BG, Zidx, Iter].
    CRLB : (5, N) float32
        CRLB diagonal terms.
    LL : (N,) float32
        Log-likelihood.
    N : int
        Number of fitted ROIs.
    """
    rois_nyx = np.asarray(rois_nyx, dtype=np.float32)

    if rois_nyx.ndim != 3:
        raise ValueError(f"rois_nyx must have shape (N,Y,X), got {rois_nyx.shape}")

    N, Ysz, Xsz = map(int, rois_nyx.shape)

    if N == 0:
        K = 6
        return (
            np.zeros((K, 0), dtype=np.float32),
            np.zeros((K - 1, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            0,
        )

    rois_loclib_nxy = rois_nyx_image_to_loclib_nxy(rois_nyx)

    em = np.float32(EMexcessNoise if EMexcessNoise else 1.0)
    rois_in = np.ascontiguousarray(
        rois_loclib_nxy.astype(np.float32, copy=False) / em,
        dtype=np.float32,
    )

    if zseed_abs is None:
        zseed_abs = zseed_center_to_absolute(coeff_loclib_single, 0.0)

    if isinstance(varstack, (int, float)) and varstack == 0:
        var_arg = np.array(0, dtype=np.float32)
    else:
        var_arg = np.asarray(varstack, dtype=np.float32)
        if var_arg.shape != (Ysz, Xsz, N):
            raise ValueError(f"varstack must be (Y,X,N), got {var_arg.shape}")

    datasize = np.array([Ysz, Xsz, N], dtype=np.int32)

    K = 6
    P_out = (
        prealloc_P
        if (prealloc_P is not None and prealloc_P.shape == (K, N))
        else np.zeros((K, N), dtype=np.float32)
    )
    CRLB_out = (
        prealloc_CRLB
        if (prealloc_CRLB is not None and prealloc_CRLB.shape == (K - 1, N))
        else np.zeros((K - 1, N), dtype=np.float32)
    )
    LL_out = (
        prealloc_LogL
        if (prealloc_LogL is not None and prealloc_LogL.shape == (N,))
        else np.zeros((N,), dtype=np.float32)
    )

    fittype = np.int32(5)
    iters = np.int32(int(iterations) if iterations > 0 else 30)

    logger.debug(
        "Single-channel GPU call: N=%d Y=%d X=%d | iters=%d | zseed_abs=%.3f",
        N,
        Ysz,
        Xsz,
        int(iters),
        float(zseed_abs),
    )

    assert rois_in.dtype == np.float32 and rois_in.flags.c_contiguous
    assert coeff_loclib_single.dtype == np.float32 and coeff_loclib_single.flags.c_contiguous

    dll._mleFit(
        rois_in,
        fittype,
        iters,
        coeff_loclib_single.astype(np.float32, copy=False),
        var_arg,
        np.float32(zseed_abs),
        datasize,
        splinesize_loclib_single,
        P_out,
        CRLB_out,
        LL_out,
    )

    return P_out, CRLB_out, LL_out, N


def mle_spline_dual_channel(
    dll,
    rois_R: np.ndarray,          # image-order (N, Y, X) float32
    rois_T: np.ndarray,          # image-order (N, Y, X) float32
    dts: np.ndarray,             # (N, 4, 5) float32 for 2 channels
    shared: np.ndarray,          # (5,) or (N, 5) int32
    coeff_loclib_cbzyx: np.ndarray,
    splinesize_loclib_xyzbc: np.ndarray,
    zseed_abs: np.ndarray,       # (N,) float32 absolute spline z seeds
    iterations: int = 30,
    varim: int | np.ndarray = 0,
    prealloc_P: np.ndarray | None = None,
    prealloc_CRLB: np.ndarray | None = None,
    prealloc_LogL: np.ndarray | None = None,
):
    """
    Low-level wrapper around the loclib multichannel spline MLE CUDA routine.

    This is the dual-channel counterpart to mle_spline_single_channel(). It packs
    paired R/T ROIs into loclib's multichannel layout, normalizes the shared-mask
    and dTS arrays, allocates or reuses output buffers, calls `_mleFit_MultiChannel`,
    and returns raw P/CRLB/LL arrays.

    It does not decide whether the model is free-ratio or fixed-ratio. That choice
    is encoded upstream through the shared mask and dTS array.

    Parameters
    ----------
    dll
        localizationlib(...) instance exposing `_mleFit_MultiChannel`.
    rois_R, rois_T
        Paired image-order ROIs, shape (N, Y, X), float32 photons.
        These are packed internally into loclib layout (C, N, X, Y).
    dts
        Per-fit subpixel coordinate offsets and multiplicative scales, shape (N, 4, 5):
            dts[:, 0, :] -> parameter offsets (xy) for channel R
            dts[:, 1, :] -> parameter offsets (xy) for channel T
            dts[:, 2, :] -> multiplicative scales for channel R
            dts[:, 3, :] -> multiplicative scales for channel T
    shared
        Shared-mask over base params [y, x, z, photons, bg].
        Either shape (5,) or (N, 5).
    coeff_loclib_cbzyx
        Packed multichannel spline coefficients with shape
        (channel, basis64, z, y, x), C-contiguous. Its flattened memory matches
        the multichannel loclib CUDA convention:

            coeff(x, y, z, basis64, channel)

    splinesize_loclib_xyzbc
        int32 metadata equivalent to MATLAB size(coeff):
        [x, y, z, basis64, channel].
    zseed_abs
        Initial z seeds, shape (N,), float32.
    iterations
        LM iteration cap.
    varim
        0 for EMCCD-like mode, or per-pixel variance array matching the packed
        multichannel data layout expected by the DLL.
    prealloc_P, prealloc_CRLB, prealloc_LogL
        Optional preallocated output arrays.

    Returns
    -------
    P : (NV+1, N) float32
        Fit parameters + iteration count row.
    CRLB : (NV, N) float32
        CRLB diagonal terms.
    LL : (N,) float32
        Log-likelihood.
    N : int
        Number of fitted paired ROIs.
    """

    # ----------------------------
    # Validate ROI arrays
    # ----------------------------
    rois_R = np.asarray(rois_R, dtype=np.float32)
    rois_T = np.asarray(rois_T, dtype=np.float32)

    if rois_R.ndim != 3 or rois_T.ndim != 3:
        raise ValueError(
            f"rois_R and rois_T must both be 3D (N,K,K); "
            f"got {rois_R.ndim}D and {rois_T.ndim}D"
        )
    if rois_R.shape != rois_T.shape:
        raise ValueError(f"ROI shape mismatch: {rois_R.shape} vs {rois_T.shape}")

    N, K1, K2 = map(int, rois_R.shape)
    if K1 != K2:
        raise ValueError(f"ROIs must be square, got {rois_R.shape}")

    if N == 0:
        # Need shared to determine NV if possible
        shared_arr = np.asarray(shared, dtype=np.int32)
        if shared_arr.shape == (5,):
            sum_shared = int(np.sum(shared_arr))
        elif shared_arr.ndim == 2 and shared_arr.shape[1] == 5:
            sum_shared = int(np.sum(shared_arr[0])) if shared_arr.shape[0] > 0 else 0
        else:
            raise ValueError(f"shared must be (5,) or (N,5), got {shared_arr.shape}")

        n_channels = 2
        NV = int(5 * n_channels - sum_shared * (n_channels - 1))
        return (
            np.zeros((NV + 1, 0), dtype=np.float32),
            np.zeros((NV, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            0,
        )

    K = K1

    # ----------------------------
    # Pack multichannel ROI tensor for loclib.
    #
    # Python/preprocessing gives image-order ROIs:
    #     rois_R, rois_T: (N, Y, X)
    #
    # The loclib CUDA convention is:
    #     data(x, y, fit, channel)
    #
    # pack_dual_rois_nyx_image_to_loclib_cnxy() returns a C-contiguous
    # NumPy array with shape (C, N, X, Y) whose flattened memory matches
    # that MATLAB column-major layout.
    # ----------------------------
    data = pack_dual_rois_nyx_image_to_loclib_cnxy(rois_R, rois_T)

    # ----------------------------
    # Normalize shared mask
    # DLL expects (N,5)
    # ----------------------------
    shared = np.asarray(shared, dtype=np.int32)
    if shared.shape == (5,):
        sharedA = np.repeat(shared[None, :], N, axis=0)
    elif shared.shape == (N, 5):
        sharedA = shared
    else:
        raise ValueError(f"shared must be (5,) or ({N},5), got {shared.shape}")
    sharedA = np.ascontiguousarray(sharedA, dtype=np.int32)

    # Number of reduced fit parameters after applying sharing
    sum_shared = int(np.sum(sharedA[0]))
    n_channels = 2
    NV = int(5 * n_channels - sum_shared * (n_channels - 1))

    # ----------------------------
    # Validate dTS
    # ----------------------------
    dts = np.asarray(dts, dtype=np.float32)
    expected_dts_shape = (N, 2 * n_channels, 5)
    if dts.shape != expected_dts_shape:
        raise ValueError(f"dts must be {expected_dts_shape}, got {dts.shape}")
    dts = np.ascontiguousarray(dts, dtype=np.float32)

    # ----------------------------
    # Validate z seeds
    # ----------------------------
    zseed_abs = np.asarray(zseed_abs, dtype=np.float32).reshape(-1)
    if zseed_abs.shape != (N,):
        raise ValueError(f"zseed_abs must be ({N},), got {zseed_abs.shape}")
    zseed_abs = np.ascontiguousarray(zseed_abs, dtype=np.float32)

    # ----------------------------
    # Validate coeff + splinesize.
    # We do not reinterpret layout here; caller must prepare loclib memory.
    # ----------------------------
    coeff_loclib_cbzyx = np.ascontiguousarray(
        np.asarray(coeff_loclib_cbzyx, dtype=np.float32)
    )
    splinesize_loclib_xyzbc = np.ascontiguousarray(
        np.asarray(splinesize_loclib_xyzbc, dtype=np.int32)
    )

    # ----------------------------
    # datasize expected by the multichannel DLL
    # loclib metadata follows the MATLAB logical order:
    #     data(x, y, fit, channel)
    #
    # data is stored in NumPy as C-contiguous (C, N, X, Y), so flipping the shape
    # gives the metadata order expected by the DLL:
    #     [Y, X, N, C]
    # Since W == H for square ROIs, datasize values are unchanged, but memory order matters.
    # ----------------------------
    datasize = np.array(np.flip(data.shape), dtype=np.int32)

    # ----------------------------
    # Variance argument
    # Variance-map convention:
    #   - scalar 0 selects the EMCCD-like path
    #   - full variance arrays must match packed data shape and memory order
    # ----------------------------
    if isinstance(varim, (int, float)) and varim == 0:
        var_arg = np.array(0, dtype=np.float32)
    else:
        var_arg = np.asarray(varim, dtype=np.float32)
        if var_arg.shape != data.shape:
            raise ValueError(
                f"varim must match packed data shape {data.shape}, got {var_arg.shape}"
            )
        var_arg = np.ascontiguousarray(var_arg, dtype=np.float32)

    # ----------------------------
    # Output buffers
    # P has NV+1 rows because last row stores iteration count
    # ----------------------------
    P = (
        prealloc_P
        if (prealloc_P is not None and prealloc_P.shape == (NV + 1, N))
        else np.zeros((NV + 1, N), dtype=np.float32)
    )
    CRLB = (
        prealloc_CRLB
        if (prealloc_CRLB is not None and prealloc_CRLB.shape == (NV, N))
        else np.zeros((NV, N), dtype=np.float32)
    )
    LL = (
        prealloc_LogL
        if (prealloc_LogL is not None and prealloc_LogL.shape == (N,))
        else np.zeros((N,), dtype=np.float32)
    )

    P = np.ascontiguousarray(P, dtype=np.float32)
    CRLB = np.ascontiguousarray(CRLB, dtype=np.float32)
    LL = np.ascontiguousarray(LL, dtype=np.float32)

    # ----------------------------
    # DLL call
    # fittype = 2 => spline
    # ----------------------------
    dll._mleFit_MultiChannel(
        data,
        np.int32(2),                 # spline fit
        sharedA,
        np.int32(int(iterations)),
        coeff_loclib_cbzyx,
        dts,
        var_arg,
        zseed_abs,
        datasize,
        splinesize_loclib_xyzbc,
        P,
        CRLB,
        LL,
    )

    return P, CRLB, LL, N


# =============================================================================
# Localization result formatting
# =============================================================================
def format_single_channel_fit_results(
    P: np.ndarray, CRLB: np.ndarray, LogL: np.ndarray,
    xpix_N: np.ndarray, ypix_N: np.ndarray, frame_N: np.ndarray,
    *,
    roi_size: int,
    dz_nm: float, z0_index: int,
    mirror: bool | int, normf: float,
    EMexcessNoise: int = 1, RI_mismatch: float = 1.0,
    pixelsize_nm: float = 100.0,
) -> dict[str, np.ndarray]:
    
    """
    Format raw single-channel spline GPU outputs into export-ready columns.

    This function converts raw P/CRLB/LogL arrays from the single-channel spline
    fitter into full-frame pixel coordinates, nanometer coordinates, photon/bg
    estimates, localization precision fields, and quality-control flags.

    It does not filter, group, or write files.
    """

    Yc   = P[0].astype(np.float32, copy=False)
    Xc   = P[1].astype(np.float32, copy=False)
    Phot = P[2].astype(np.float32, copy=False)
    BG   = P[3].astype(np.float32, copy=False)
    Zidx = P[4].astype(np.float32, copy=False)
    Iter = P[5].astype(np.float32, copy=False)

    half   = np.float32((roi_size - 1) // 2)
    seed_x = xpix_N.astype(np.float32, copy=False)
    seed_y = ypix_N.astype(np.float32, copy=False)

    xpix_roi = Xc
    ypix_roi = Yc

    # Edge clamp flag per CUDA bounds: center ± roi_size/4
    center    = np.float32((roi_size - 1) / 2.0)
    halfwidth = np.float32(roi_size / 4.0)
    lo = center - halfwidth
    hi = center + halfwidth
    eps = np.float32(1e-3)
    x_edge = (np.abs(Xc - lo) <= eps) | (np.abs(Xc - hi) <= eps)
    y_edge = (np.abs(Yc - lo) <= eps) | (np.abs(Yc - hi) <= eps)
    edge_clamped = (x_edge | y_edge)

    if bool(mirror):
        xcam = (half - Xc) + seed_x
    else:
        xcam = (Xc - half) + seed_x
    ycam = (Yc - half) + seed_y

    photons = Phot * np.float32(EMexcessNoise) * np.float32(normf)
    bg      = BG   * np.float32(EMexcessNoise)
    znm     = -((Zidx - np.float32(z0_index)) * np.float32(dz_nm)) * np.float32(RI_mismatch)

    var_x_pix = CRLB[1].astype(np.float32, copy=False)
    var_y_pix = CRLB[0].astype(np.float32, copy=False)
    locprec_pix = np.sqrt(0.5 * (var_x_pix + var_y_pix))
    locprecnm = locprec_pix * np.float32(pixelsize_nm)
    var_z_idx = CRLB[4].astype(np.float32, copy=False)
    locprecznm = np.sqrt(var_z_idx) * np.float32(dz_nm) * np.float32(RI_mismatch)

    px_nm = np.float32(pixelsize_nm)
    xnm = (xcam + np.float32(1.0)) * px_nm
    ynm = (ycam + np.float32(1.0)) * px_nm

    return dict(
        frame   = frame_N.astype(np.int32, copy=False),
        xpix_seed = seed_x.astype(np.int32, copy=False),
        ypix_seed = seed_y.astype(np.int32, copy=False),        
        xnm     = xnm,
        ynm     = ynm,
        xpix_roi= xpix_roi,
        ypix_roi= ypix_roi,
        xpix    = xcam,
        ypix    = ycam,
        photons = photons,
        bg      = bg,
        znm     = znm,
        zidx    = Zidx,
        xerr    = np.sqrt(CRLB[1]).astype(np.float32, copy=False),
        yerr    = np.sqrt(CRLB[0]).astype(np.float32, copy=False),
        photerr = (np.sqrt(CRLB[2]) * np.float32(EMexcessNoise)).astype(np.float32, copy=False),
        bgerr   = (np.sqrt(CRLB[3]) * np.float32(EMexcessNoise)).astype(np.float32, copy=False),
        zerr    = (np.sqrt(CRLB[4]) * np.float32(dz_nm) * np.float32(RI_mismatch)).astype(np.float32, copy=False),
        logL    = LogL.astype(np.float32, copy=False),
        locprecnm = locprecnm,
        locprecznm = locprecznm,
        iter    = Iter,
        edge_clamped = edge_clamped.astype(np.bool_, copy=False),
    )


def format_global_fit_results_free_ratio(
    P: np.ndarray,
    CRLB: np.ndarray,
    LL: np.ndarray,
    R_centers: np.ndarray,       # (N,2) [x,y] integer ROI centers
    T_centers: np.ndarray,       # (N,2) [x,y] integer ROI centers
    R_dxdy: np.ndarray,          # (N,2) [dx,dy]
    T_dxdy: np.ndarray,          # (N,2) [dx,dy]
    frames: np.ndarray,          # (N,)
    *,
    roi_size: int,
    dz_nm: float,
    z0_index: int,
    normf: np.ndarray | None = None,
    EMexcessNoise: int = 1,
    RI_mismatch: float = 1.0,
    pixelsize_nm: float = 100.0,
    main_channel: str = "R",     # "R", "T", or "mean"
    free_ratio_output_channel: int = 0,
) -> dict[str, np.ndarray]:
    """
    Format raw free-ratio global GPU fit outputs into export-ready localization columns.

    This function converts raw P/CRLB/LL arrays from the GPU fitter into the
    SMAP-like columns used by downstream filtering, grouping, and CSV export.
    It does not filter, group, or write files.

    Free-ratio shared model:
    shared = [1, 1, 1, 0, 0]
                y  x  z  I  bg

    Free-ratio P rows:
        0 y_shared_base
        1 x_shared_base
        2 z_index
        3 photons_R_raw
        4 photons_T_raw
        5 bg_R
        6 bg_T
        7 iterations

    CRLB rows:
        0 y
        1 x
        2 z
        3 phot_R
        4 phot_T
        5 bg_R
        6 bg_T
    """

    P = np.asarray(P, dtype=np.float32)
    CRLB = np.asarray(CRLB, dtype=np.float32)
    LL = np.asarray(LL, dtype=np.float32)

    CRLB = np.nan_to_num(CRLB, nan=0.0)
    CRLB = np.maximum(CRLB, 0.0)
    LL = np.nan_to_num(LL, nan=0.0)

    R_centers = np.asarray(R_centers)
    T_centers = np.asarray(T_centers)
    R_dxdy = np.asarray(R_dxdy, dtype=np.float32)
    T_dxdy = np.asarray(T_dxdy, dtype=np.float32)
    frames = np.asarray(frames)

    N = P.shape[1]
    if P.shape[0] != 8:
        raise ValueError(f"Expected free-ratio P shape (8,N), got {P.shape}")
    if CRLB.shape != (7, N):
        raise ValueError(f"Expected free-ratio CRLB shape (7,N), got {CRLB.shape}")
    if LL.shape != (N,):
        raise ValueError(f"Expected LL shape ({N},), got {LL.shape}")

    for name, arr, shape in [
        ("R_centers", R_centers, (N, 2)),
        ("T_centers", T_centers, (N, 2)),
        ("R_dxdy", R_dxdy, (N, 2)),
        ("T_dxdy", T_dxdy, (N, 2)),
    ]:
        if arr.shape != shape:
            raise ValueError(f"{name} must be {shape}, got {arr.shape}")

    if frames.shape != (N,):
        raise ValueError(f"frames must be ({N},), got {frames.shape}")

    if normf is None:
        normf_arr = np.ones(2, dtype=np.float32)
    else:
        normf_arr = np.asarray(normf, dtype=np.float32).reshape(-1)
        if normf_arr.size < 2:
            normf_arr = np.array([float(normf_arr[0]), 1.0], dtype=np.float32)
        else:
            normf_arr = normf_arr[:2]

    if main_channel not in {"R", "T", "mean"}:
        raise ValueError("main_channel must be 'R', 'T', or 'mean'")

    # Shared base coordinates from fitter
    y_base_local = P[0]
    x_base_local = P[1]
    z_idx = P[2]

    # Add independent per-channel residuals
    x_R_local = x_base_local + R_dxdy[:, 0]
    y_R_local = y_base_local + R_dxdy[:, 1]
    x_T_local = x_base_local + T_dxdy[:, 0]
    y_T_local = y_base_local + T_dxdy[:, 1]

    # Fold fitted local ROI coordinates back into full-frame pixel coordinates
    # using the same ROI-center convention as the Stage 1 formatter.
    half = np.float32((roi_size - 1) // 2)

    x_R_pix = R_centers[:, 0].astype(np.float64) + (x_R_local.astype(np.float64) - float(half))
    y_R_pix = R_centers[:, 1].astype(np.float64) + (y_R_local.astype(np.float64) - float(half))
    x_T_pix = T_centers[:, 0].astype(np.float64) + (x_T_local.astype(np.float64) - float(half))
    y_T_pix = T_centers[:, 1].astype(np.float64) + (y_T_local.astype(np.float64) - float(half))

    # Primary reported global coordinates are in R/reference frame
    px_nm = np.float32(pixelsize_nm)
    x_nm = (x_R_pix + 1.0) * float(px_nm)
    y_nm = (y_R_pix + 1.0) * float(px_nm)
    z_nm = -((z_idx - np.float32(z0_index)) * np.float32(dz_nm)) * np.float32(RI_mismatch)

    # Photometry
    phot1 = P[3] * np.float32(EMexcessNoise) * normf_arr[0]
    phot2 = P[4] * np.float32(EMexcessNoise) * normf_arr[1]
    photons_total = phot1 + phot2

    ratio_T_over_total = np.divide(
        phot2,
        photons_total,
        out=np.full_like(phot2, np.nan, dtype=np.float32),
        where=(photons_total > 0),
    )
    ratio_T_over_R = np.divide(
        phot2,
        phot1,
        out=np.full_like(phot2, np.nan, dtype=np.float32),
        where=(phot1 > 0),
    )

    bg1 = P[5] * np.float32(EMexcessNoise)
    bg2 = P[6] * np.float32(EMexcessNoise)

    # Errors from CRLB
    xpixerr = np.sqrt(np.maximum(CRLB[1], 0)).astype(np.float32)
    ypixerr = np.sqrt(np.maximum(CRLB[0], 0)).astype(np.float32)
    znmerr = (
        np.sqrt(np.maximum(CRLB[2], 0)).astype(np.float32)
        * np.float32(dz_nm)
        * np.float32(RI_mismatch)
    )

    phot1err = (
        np.sqrt(np.maximum(CRLB[3], 0)).astype(np.float32)
        * np.float32(EMexcessNoise)
        * normf_arr[0]
    )
    phot2err = (
        np.sqrt(np.maximum(CRLB[4], 0)).astype(np.float32)
        * np.float32(EMexcessNoise)
        * normf_arr[1]
    )
    bg1err = (
        np.sqrt(np.maximum(CRLB[5], 0)).astype(np.float32)
        * np.float32(EMexcessNoise)
    )
    bg2err = (
        np.sqrt(np.maximum(CRLB[6], 0)).astype(np.float32)
        * np.float32(EMexcessNoise)
    )

    # XY precision
    locprecnm = (
        np.sqrt(0.5 * (np.maximum(CRLB[0], 0) + np.maximum(CRLB[1], 0)))
        * np.float32(pixelsize_nm)
    ).astype(np.float32)

    # MATLAB-like locprecznm
    locprecznm = znmerr.astype(np.float32)

    # nm-domain errors
    xnmerr = xpixerr * np.float32(pixelsize_nm)
    ynmerr = ypixerr * np.float32(pixelsize_nm)

    # "phot" / "photerr" follow MATLAB mainchannel logic
    if main_channel == "R":
        phot = phot1.astype(np.float32)
        photerr = phot1err.astype(np.float32)
    elif main_channel == "T":
        phot = phot2.astype(np.float32)
        photerr = phot2err.astype(np.float32)
    else:  # "mean"
        # inverse-variance weighted average
        var1 = np.maximum(phot1err.astype(np.float32) ** 2, 1e-12)
        var2 = np.maximum(phot2err.astype(np.float32) ** 2, 1e-12)
        w1 = 1.0 / var1
        w2 = 1.0 / var2
        phot = ((phot1 * w1) + (phot2 * w2)) / (w1 + w2)
        photerr = np.sqrt(1.0 / (w1 + w2)).astype(np.float32)

    # MATLAB-like convenience fields
    channel = np.full((N,), int(free_ratio_output_channel), dtype=np.int32)
    iterations = P[7].astype(np.float32)
    
    # Exported logLikelihood is not the raw multichannel DLL output.
    # For the standard 2-camera system with equal
    # channel weighting [1, 1], this is simply a divide-by-2 normalization.
    logLikelihood = (LL.astype(np.float32) / np.float32(2.0)).astype(np.float32)

    # MATLAB-style relative likelihood statistic used in downstream tables.
    LLrel = compute_llrel(
        logLikelihood,
        roi_size=roi_size,
        num_channels=2,
        em_on=bool(EMexcessNoise == 2),
    )

    return dict(
        frame=frames.astype(np.int32, copy=False),

        **{"x [nm]": x_nm},
        **{"y [nm]": y_nm},
        **{"z [nm]": z_nm.astype(np.float32)},

        locprecnm=locprecnm,
        locprecznm=locprecznm,

        channel=channel,

        bg1=bg1.astype(np.float32),
        bg1err=bg1err.astype(np.float32),
        bg2=bg2.astype(np.float32),
        bg2err=bg2err.astype(np.float32),

        iterations=iterations,
        logLikelihood=logLikelihood,
        LLrel=LLrel,

        phot=phot.astype(np.float32),
        photerr=photerr.astype(np.float32),

        phot1=phot1.astype(np.float32),
        phot1err=phot1err.astype(np.float32),
        phot2=phot2.astype(np.float32),
        phot2err=phot2err.astype(np.float32),

        photons_total=photons_total.astype(np.float32),
        ratio_T_over_R=ratio_T_over_R.astype(np.float32),
        ratio_T_over_total=ratio_T_over_total.astype(np.float32),

        xpix=np.asarray(x_R_pix, dtype=np.float64),
        ypix=np.asarray(y_R_pix, dtype=np.float64),
        xpixerr=xpixerr,
        ypixerr=ypixerr,

        xnmerr=xnmerr.astype(np.float32),
        ynmerr=ynmerr.astype(np.float32),
        znmerr=znmerr.astype(np.float32),

        xpix_roi_center_R=R_centers[:, 0].astype(np.int32),
        ypix_roi_center_R=R_centers[:, 1].astype(np.int32),
        xpix_roi_center_T=T_centers[:, 0].astype(np.int32),
        ypix_roi_center_T=T_centers[:, 1].astype(np.int32),
    )


def format_global_fit_results_fixed_ratio(
    P: np.ndarray,
    CRLB: np.ndarray,
    LL: np.ndarray,
    R_centers: np.ndarray,       # (N,2) [x,y] integer ROI centers
    T_centers: np.ndarray,       # (N,2) [x,y] integer ROI centers
    R_dxdy: np.ndarray,          # (N,2) [dx,dy]
    T_dxdy: np.ndarray,          # (N,2) [dx,dy]
    frames: np.ndarray,          # (N,)
    *,
    assigned_ratio: np.ndarray,   # (N,) final T/R photon ratio
    assigned_channel: np.ndarray, # (N,) class index, 1..len(fixed_ratios)
    ll_second: np.ndarray,        # (N,) raw second-best LL from DLL
    roi_size: int,
    dz_nm: float,
    z0_index: int,
    normf: np.ndarray | None = None,
    EMexcessNoise: int = 1,
    RI_mismatch: float = 1.0,
    pixelsize_nm: float = 100.0,
    main_channel: str = "R",      # "R", "T", or "mean"
) -> dict[str, np.ndarray]:
    """
    Format raw fixed-ratio global GPU fit outputs into export-ready localization columns.

    This function converts raw P/CRLB/LL arrays from the selected fixed-ratio GPU
    fit into the SMAP-like columns used by downstream filtering, grouping, and CSV
    export. It also records the assigned ratio/channel and likelihood margin fields.

    It does not filter, group, or write files.

    Fixed-ratio shared model:
        shared = [1, 1, 1, 1, 0]
                  y  x  z  I  bg

    P rows:
        0 y_shared_base
        1 x_shared_base
        2 z_index
        3 photons_shared_raw
        4 bg_R
        5 bg_T
        6 iterations

    CRLB rows:
        0 y
        1 x
        2 z
        3 photons_shared
        4 bg_R
        5 bg_T
    """

    P = np.asarray(P, dtype=np.float32)
    CRLB = np.asarray(CRLB, dtype=np.float32)
    LL = np.asarray(LL, dtype=np.float32)

    CRLB = np.nan_to_num(CRLB, nan=0.0)
    CRLB = np.maximum(CRLB, 0.0)
    LL = np.nan_to_num(LL, nan=0.0)

    R_centers = np.asarray(R_centers)
    T_centers = np.asarray(T_centers)
    R_dxdy = np.asarray(R_dxdy, dtype=np.float32)
    T_dxdy = np.asarray(T_dxdy, dtype=np.float32)
    frames = np.asarray(frames)

    assigned_ratio = np.asarray(assigned_ratio, dtype=np.float32).reshape(-1)
    assigned_channel = np.asarray(assigned_channel, dtype=np.int32).reshape(-1)
    ll_second = np.asarray(ll_second, dtype=np.float32).reshape(-1)

    N = P.shape[1]
    if P.shape[0] != 7:
        raise ValueError(f"Expected fixed-ratio P shape (7,N), got {P.shape}")
    if CRLB.shape != (6, N):
        raise ValueError(f"Expected fixed-ratio CRLB shape (6,N), got {CRLB.shape}")
    if LL.shape != (N,):
        raise ValueError(f"Expected LL shape ({N},), got {LL.shape}")
    if assigned_ratio.shape != (N,):
        raise ValueError(f"assigned_ratio must be ({N},), got {assigned_ratio.shape}")
    if assigned_channel.shape != (N,):
        raise ValueError(f"assigned_channel must be ({N},), got {assigned_channel.shape}")
    if ll_second.shape != (N,):
        raise ValueError(f"ll_second must be ({N},), got {ll_second.shape}")

    for name, arr, shape in [
        ("R_centers", R_centers, (N, 2)),
        ("T_centers", T_centers, (N, 2)),
        ("R_dxdy", R_dxdy, (N, 2)),
        ("T_dxdy", T_dxdy, (N, 2)),
    ]:
        if arr.shape != shape:
            raise ValueError(f"{name} must be {shape}, got {arr.shape}")

    if frames.shape != (N,):
        raise ValueError(f"frames must be ({N},), got {frames.shape}")

    if normf is None:
        normf_arr = np.ones(2, dtype=np.float32)
    else:
        normf_arr = np.asarray(normf, dtype=np.float32).reshape(-1)
        if normf_arr.size < 2:
            normf_arr = np.array([float(normf_arr[0]), 1.0], dtype=np.float32)
        else:
            normf_arr = normf_arr[:2]

    if main_channel not in {"R", "T", "mean"}:
        raise ValueError("main_channel must be 'R', 'T', or 'mean'")

    # Shared base coordinates from fitter
    y_base_local = P[0]
    x_base_local = P[1]
    z_idx = P[2]

    # Add independent per-channel residuals
    x_R_local = x_base_local + R_dxdy[:, 0]
    y_R_local = y_base_local + R_dxdy[:, 1]
    x_T_local = x_base_local + T_dxdy[:, 0]
    y_T_local = y_base_local + T_dxdy[:, 1]

    # Fold fitted local ROI coordinates back into full-frame pixel coordinates
    # using the same ROI-center convention as the Stage 1 formatter.
    half = np.float32((roi_size - 1) // 2)

    x_R_pix = R_centers[:, 0].astype(np.float64) + (x_R_local.astype(np.float64) - float(half))
    y_R_pix = R_centers[:, 1].astype(np.float64) + (y_R_local.astype(np.float64) - float(half))
    x_T_pix = T_centers[:, 0].astype(np.float64) + (x_T_local.astype(np.float64) - float(half))
    y_T_pix = T_centers[:, 1].astype(np.float64) + (y_T_local.astype(np.float64) - float(half))

    # Report coordinates in R/reference frame
    px_nm = np.float32(pixelsize_nm)
    x_nm = (x_R_pix + 1.0) * float(px_nm)
    y_nm = (y_R_pix + 1.0) * float(px_nm)
    z_nm = -((z_idx - np.float32(z0_index)) * np.float32(dz_nm)) * np.float32(RI_mismatch)

    # Shared photometry
    phot_shared = P[3] * np.float32(EMexcessNoise) * normf_arr[0]
    phot1 = phot_shared.astype(np.float32)
    phot2 = (phot_shared * assigned_ratio).astype(np.float32)
    photons_total = (phot1 + phot2).astype(np.float32)

    ratio_T_over_R = assigned_ratio.astype(np.float32)
    ratio_T_over_total = np.divide(
        assigned_ratio,
        1.0 + assigned_ratio,
        out=np.full_like(assigned_ratio, np.nan, dtype=np.float32),
        where=np.isfinite(assigned_ratio),
    ).astype(np.float32)

    bg1 = P[4] * np.float32(EMexcessNoise)
    bg2 = P[5] * np.float32(EMexcessNoise)

    # Errors from CRLB
    xpixerr = np.sqrt(np.maximum(CRLB[1], 0)).astype(np.float32)
    ypixerr = np.sqrt(np.maximum(CRLB[0], 0)).astype(np.float32)
    znmerr = (
        np.sqrt(np.maximum(CRLB[2], 0)).astype(np.float32)
        * np.float32(dz_nm)
        * np.float32(RI_mismatch)
    )

    phot_shared_err = (
        np.sqrt(np.maximum(CRLB[3], 0)).astype(np.float32)
        * np.float32(EMexcessNoise)
        * normf_arr[0]
    )
    phot1err = phot_shared_err.astype(np.float32)
    phot2err = (phot_shared_err * assigned_ratio).astype(np.float32)

    bg1err = (
        np.sqrt(np.maximum(CRLB[4], 0)).astype(np.float32)
        * np.float32(EMexcessNoise)
    )
    bg2err = (
        np.sqrt(np.maximum(CRLB[5], 0)).astype(np.float32)
        * np.float32(EMexcessNoise)
    )

    # XY precision
    locprecnm = (
        np.sqrt(0.5 * (np.maximum(CRLB[0], 0) + np.maximum(CRLB[1], 0)))
        * np.float32(pixelsize_nm)
    ).astype(np.float32)

    locprecznm = znmerr.astype(np.float32)

    xnmerr = xpixerr * np.float32(pixelsize_nm)
    ynmerr = ypixerr * np.float32(pixelsize_nm)

    if main_channel == "R":
        phot = phot1.astype(np.float32)
        photerr = phot1err.astype(np.float32)
    elif main_channel == "T":
        phot = phot2.astype(np.float32)
        photerr = phot2err.astype(np.float32)
    else:  # "mean"
        var1 = np.maximum(phot1err.astype(np.float32) ** 2, 1e-12)
        var2 = np.maximum(phot2err.astype(np.float32) ** 2, 1e-12)
        w1 = 1.0 / var1
        w2 = 1.0 / var2
        phot = ((phot1 * w1) + (phot2 * w2)) / (w1 + w2)
        photerr = np.sqrt(1.0 / (w1 + w2)).astype(np.float32)

    channel = assigned_channel.astype(np.int32, copy=False)
    iterations = P[6].astype(np.float32)

    # Match exported convention used elsewhere in pipeline
    logLikelihood = (LL.astype(np.float32) / np.float32(2.0)).astype(np.float32)
    fixed_ratio_ll_second = (ll_second.astype(np.float32) / np.float32(2.0)).astype(np.float32)
    fixed_ratio_ll_margin = (logLikelihood - fixed_ratio_ll_second).astype(np.float32)

    LLrel = compute_llrel(
        logLikelihood,
        roi_size=roi_size,
        num_channels=2,
        em_on=bool(EMexcessNoise == 2),
    )

    return dict(
        frame=frames.astype(np.int32, copy=False),

        **{"x [nm]": x_nm},
        **{"y [nm]": y_nm},
        **{"z [nm]": z_nm.astype(np.float32)},

        locprecnm=locprecnm,
        locprecznm=locprecznm,

        channel=channel,
        assigned_ratio=assigned_ratio.astype(np.float32),

        bg1=bg1.astype(np.float32),
        bg1err=bg1err.astype(np.float32),
        bg2=bg2.astype(np.float32),
        bg2err=bg2err.astype(np.float32),

        iterations=iterations,
        logLikelihood=logLikelihood,
        LLrel=LLrel,
        fixed_ratio_ll_best=logLikelihood,
        fixed_ratio_ll_second=fixed_ratio_ll_second,
        fixed_ratio_ll_margin=fixed_ratio_ll_margin,

        phot=phot.astype(np.float32),
        photerr=photerr.astype(np.float32),

        phot1=phot1.astype(np.float32),
        phot1err=phot1err.astype(np.float32),
        phot2=phot2.astype(np.float32),
        phot2err=phot2err.astype(np.float32),

        photons_total=photons_total.astype(np.float32),
        ratio_T_over_R=ratio_T_over_R.astype(np.float32),
        ratio_T_over_total=ratio_T_over_total.astype(np.float32),

        xpix=np.asarray(x_R_pix, dtype=np.float64),
        ypix=np.asarray(y_R_pix, dtype=np.float64),
        xpixerr=xpixerr,
        ypixerr=ypixerr,

        xnmerr=xnmerr.astype(np.float32),
        ynmerr=ynmerr.astype(np.float32),
        znmerr=znmerr.astype(np.float32),

        xpix_roi_center_R=R_centers[:, 0].astype(np.int32),
        ypix_roi_center_R=R_centers[:, 1].astype(np.int32),
        xpix_roi_center_T=T_centers[:, 0].astype(np.int32),
        ypix_roi_center_T=T_centers[:, 1].astype(np.int32),
    )


# =============================================================================
# Stage 3 dTS builders
# =============================================================================
def build_dts_dual_channel_free_ratio(
    R_dxdy: np.ndarray,
    T_dxdy: np.ndarray,
) -> np.ndarray:
    """
    Build dTS array for Stage 3 free-ratio dual-channel global fitting.

    Parameter order used by the global fitter:
        [y, x, z, photons, bg]

    Free-ratio shared model:
        shared = [1, 1, 1, 0, 0]
                  y  x  z  I  bg

    dTS layout:
        dTS[:, 0, :] -> parameter offsets for channel R
        dTS[:, 1, :] -> parameter offsets for channel T
        dTS[:, 2, :] -> multiplicative scales for channel R
        dTS[:, 3, :] -> multiplicative scales for channel T

    R_dxdy/T_dxdy are expected as [dx, dy].
    Since parameter order is [y, x, z, photons, bg]:
        dTS[..., 0] receives dy
        dTS[..., 1] receives dx
    """
    R_dxdy = np.asarray(R_dxdy, dtype=np.float32)
    T_dxdy = np.asarray(T_dxdy, dtype=np.float32)

    if R_dxdy.ndim != 2 or R_dxdy.shape[1] != 2:
        raise ValueError(f"R_dxdy must have shape (N, 2), got {R_dxdy.shape}")
    if T_dxdy.shape != R_dxdy.shape:
        raise ValueError(f"T_dxdy must match R_dxdy, got {T_dxdy.shape} vs {R_dxdy.shape}")

    N = int(R_dxdy.shape[0])
    dTS = np.zeros((N, 4, 5), dtype=np.float32)

    # Subpixel coordinate offsets, parameter order [y, x, z, photons, bg].
    # R_dxdy/T_dxdy are residuals from each channel's rounded ROI center:
    #   dTS[..., 0] receives dy
    #   dTS[..., 1] receives dx
    dTS[:, 0, 0] = R_dxdy[:, 1]  # R y offset = R dy
    dTS[:, 0, 1] = R_dxdy[:, 0]  # R x offset = R dx

    dTS[:, 1, 0] = T_dxdy[:, 1]  # T y offset = T dy
    dTS[:, 1, 1] = T_dxdy[:, 0]  # T x offset = T dx

    # Multiplicative scales: all 1. Free photons are unshared, so photon scale is ignored.
    dTS[:, 2, :] = 1.0
    dTS[:, 3, :] = 1.0

    return np.ascontiguousarray(dTS, dtype=np.float32)


def build_dts_dual_channel_fixed_ratio(
    R_dxdy: np.ndarray,
    T_dxdy: np.ndarray,
    photon_ratio: float | np.ndarray,
) -> np.ndarray:
    """
    Build dTS array for Stage 3 fixed-ratio dual-channel global fitting.

    Fixed-ratio mode shares photon amplitude between R and T. The T-channel photon
    scale is set by photon_ratio, so the fitted shared photon parameter is linked
    across channels through dTS[:, 3, 3].
    
    Parameter order:
        [y, x, z, photons, bg]

    Fixed-ratio shared model:
        shared = [1, 1, 1, 1, 0]

    R_dxdy/T_dxdy are [dx, dy].
    dTS parameter index 0 is dy/y.
    dTS parameter index 1 is dx/x.

    photon_ratio can be:
        scalar: same ratio for all fits
        (N,): per-fit selected ratio
    """
    R_dxdy = np.asarray(R_dxdy, dtype=np.float32)
    T_dxdy = np.asarray(T_dxdy, dtype=np.float32)

    if R_dxdy.ndim != 2 or R_dxdy.shape[1] != 2:
        raise ValueError(f"R_dxdy must have shape (N, 2), got {R_dxdy.shape}")
    if T_dxdy.shape != R_dxdy.shape:
        raise ValueError(f"T_dxdy must match R_dxdy, got {T_dxdy.shape} vs {R_dxdy.shape}")

    N = int(R_dxdy.shape[0])

    ratio = np.asarray(photon_ratio, dtype=np.float32)
    if ratio.ndim == 0:
        ratio = np.full((N,), float(ratio), dtype=np.float32)
    elif ratio.shape != (N,):
        raise ValueError(f"photon_ratio must be scalar or ({N},), got {ratio.shape}")

    if not np.all(np.isfinite(ratio)):
        raise ValueError("photon_ratio contains non-finite values")
    if np.any(ratio < 0):
        raise ValueError("photon_ratio must be non-negative")

    dTS = np.zeros((N, 4, 5), dtype=np.float32)

    # Subpixel coordinate offsets, parameter order [y, x, z, photons, bg].
    # Use independent per-channel residuals from each channel's rounded ROI center.
    dTS[:, 0, 0] = R_dxdy[:, 1]  # R y offset = R dy
    dTS[:, 0, 1] = R_dxdy[:, 0]  # R x offset = R dx

    dTS[:, 1, 0] = T_dxdy[:, 1]  # T y offset = T dy
    dTS[:, 1, 1] = T_dxdy[:, 0]  # T x offset = T dx

    # Multiplicative scales
    dTS[:, 2, :] = 1.0      # R scales
    dTS[:, 3, :] = 1.0      # T scales
    dTS[:, 3, 3] = ratio    # T-channel photon scale for the shared photon parameter.

    return np.ascontiguousarray(dTS, dtype=np.float32)


# =============================================================================
# Global dual-channel spline PSF model loading
# =============================================================================
@dataclass(frozen=True)
class GlobalDualChannelSplineModel:
    """
    Global dual-channel spline model for GPUmleFit_LM_MultiChannel.

    The multichannel loclib CUDA convention for global spline coefficients is:

        coeff(x, y, z, basis64, channel)

    Python stores the same flattened memory as a C-contiguous array:

        coeff.shape == (channel, basis64, z, y, x)

    This is named coeff_loclib_cbzyx in conversion helpers.

    splinesize is the MATLAB-style metadata:

        [x, y, z, basis64, channel]
    """
    coeff: np.ndarray
    splinesize: np.ndarray
    zseed: np.float32
    dz: float
    z0: int
    normf: np.ndarray
    mirror: bool


def _mat_fields(obj: Any) -> set[str]:
    return {name for name in dir(obj) if not name.startswith("_")}


def _get_first_struct(obj: Any) -> Any:
    """
    scipy.io.loadmat may return either a mat_struct or an object ndarray of mat_structs.
    For global SXY_g in your uploaded file, it is a single mat_struct.
    """
    if isinstance(obj, np.ndarray):
        flat = np.ravel(obj)
        if flat.size == 0:
            raise RuntimeError("Empty MATLAB struct array.")
        return flat[0]
    return obj


def _extract_global_coeff_xyzw_pair(cspline: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract two channel coefficient arrays in MATLAB/SMAP layout:
        (X, Y, Z, 64)

    Handles the observed SXY_g layouts:
      1. cspline.coeff is object array/list with two arrays
      2. cspline.coeff is numeric 5D array (X,Y,Z,64,C)
      3. cspline.coeffrawref / coeffrawtar fields
      4. cspline.global.coeffrawref / coeffrawtar fields
    """

    fields = _mat_fields(cspline)

    # Case 1 / 2: cspline.coeff
    if "coeff" in fields:
        coeff = getattr(cspline, "coeff")

        # In your uploaded file:
        # coeff is ndarray shape (2,), dtype object
        if isinstance(coeff, np.ndarray) and coeff.dtype == object:
            flat = np.ravel(coeff)
            if flat.size >= 2:
                c0 = np.asarray(flat[0], dtype=np.float32)
                c1 = np.asarray(flat[1], dtype=np.float32)
                return c0, c1

        # Possible MATLAB numeric layout: (X,Y,Z,64,C)
        coeff_arr = np.asarray(coeff)
        if coeff_arr.ndim == 5 and coeff_arr.shape[3] == 64 and coeff_arr.shape[4] >= 2:
            c0 = np.asarray(coeff_arr[..., 0], dtype=np.float32)
            c1 = np.asarray(coeff_arr[..., 1], dtype=np.float32)
            return c0, c1

    # Case 3: direct fields on cspline
    if "coeffrawref" in fields and "coeffrawtar" in fields:
        c0 = np.asarray(getattr(cspline, "coeffrawref"), dtype=np.float32)
        c1 = np.asarray(getattr(cspline, "coeffrawtar"), dtype=np.float32)
        return c0, c1

    # Case 4: nested cspline.global fields
    if "global" in fields:
        g = getattr(cspline, "global")
        gfields = _mat_fields(g)
        if "coeffrawref" in gfields and "coeffrawtar" in gfields:
            c0 = np.asarray(getattr(g, "coeffrawref"), dtype=np.float32)
            c1 = np.asarray(getattr(g, "coeffrawtar"), dtype=np.float32)
            return c0, c1

    raise RuntimeError(
        "Could not find global two-channel coefficients. Expected one of: "
        "cspline.coeff{1,2}, cspline.coeff(:,:,:,:,channel), "
        "cspline.coeffrawref/coeffrawtar, or cspline.global.coeffrawref/coeffrawtar."
    )


def _validate_coeff_xyzw(coeff: np.ndarray, label: str) -> tuple[int, int, int, int]:
    coeff = np.asarray(coeff, dtype=np.float32)
    if coeff.ndim != 4:
        raise ValueError(f"{label} coeff must be 4D (X,Y,Z,64), got {coeff.shape}")
    X, Y, Z, B = map(int, coeff.shape)
    if B != 64:
        raise ValueError(f"{label} coeff last axis must be 64, got {coeff.shape}")
    if X != Y:
        raise ValueError(f"{label} coeff should be square in XY, got {X}x{Y}")
    return X, Y, Z, B


def _coeff_xyzb_to_global_channel_bzyx(coeff_xyzb: np.ndarray) -> np.ndarray:
    """
    Convert one global-fit channel coefficient block to loclib BZYX layout.

    Input logical layout:

        coeff(x, y, z, basis64), shape (X, Y, Z, B)

    Output loclib layout:

        coeff_channel_bzyx.shape == (B, Z, Y, X)

    The returned C-contiguous array is one channel block. The full global
    two-channel model stacks these blocks as:

        coeff_loclib_cbzyx.shape == (C, B, Z, Y, X)
    """
    return coeff_xyzb_matlab_to_loclib_bzyx(coeff_xyzb)


def load_global_dual_channel_spline_model(
    calib_mat_path: Path | str,
    *,
    zseed_mode: str = "z0",
    verbose: bool = True,
) -> GlobalDualChannelSplineModel:
    """
    Load the global two-channel spline PSF calibration used by Stage 3.

    The MATLAB calibration stores spline coefficients in SMAP/MATLAB-oriented
    layouts. This function extracts the reference and target channel coefficients,
    validates their shapes, converts each channel to C-order (64,Z,Y,X), stacks
    them as (C,64,Z,Y,X), and builds the splinesize metadata expected by loclib.

    Returned coeff layout:
        (C, 64, Z, Y, X)

    Returned splinesize:
        [X, Y, Z, 64, C]
    """

    src = Path(calib_mat_path)
    mat = scipy.io.loadmat(str(src), struct_as_record=False, squeeze_me=True)

    if "SXY_g" not in mat:
        raise RuntimeError(f"{src} does not contain SXY_g. This is required for global dual-channel fitting.")

    SXY_g = _get_first_struct(mat["SXY_g"])

    if not hasattr(SXY_g, "cspline"):
        raise RuntimeError("SXY_g does not contain a cspline field.")

    cspline = SXY_g.cspline

    coeff_ref_xyzw, coeff_tar_xyzw = _extract_global_coeff_xyzw_pair(cspline)

    X0, Y0, Z0, B0 = _validate_coeff_xyzw(coeff_ref_xyzw, "reference")
    X1, Y1, Z1, B1 = _validate_coeff_xyzw(coeff_tar_xyzw, "target")

    if (X0, Y0, Z0, B0) != (X1, Y1, Z1, B1):
        raise ValueError(
            "Reference and target coeff shapes do not match: "
            f"{coeff_ref_xyzw.shape} vs {coeff_tar_xyzw.shape}"
        )

    coeff_ref_channel_bzyx = _coeff_xyzb_to_global_channel_bzyx(coeff_ref_xyzw)
    coeff_tar_channel_bzyx = _coeff_xyzb_to_global_channel_bzyx(coeff_tar_xyzw)

    # Final multichannel loclib layout:
    #     coeff_loclib_cbzyx.shape == (C, B, Z, Y, X)
    #
    # Its flattened C-order memory matches MATLAB column-major:
    #     coeff(x, y, z, basis64, channel)
    coeff_loclib_cbzyx = np.ascontiguousarray(
        np.stack([coeff_ref_channel_bzyx, coeff_tar_channel_bzyx], axis=0),
        dtype=np.float32,
    )

    # DLL/MEX metadata equivalent to MATLAB size(coeff):
    #     [X, Y, Z, B, C]
    splinesize = np.asarray(np.flip(coeff_loclib_cbzyx.shape), dtype=np.int32)

    dz = float(getattr(cspline, "dz", 50.0))
    z0 = int(getattr(cspline, "z0", Z0 // 2))
    mirror = bool(getattr(cspline, "mirror", False))

    normf_raw = getattr(cspline, "normf", np.ones(2, dtype=np.float32))
    normf = np.asarray(normf_raw, dtype=np.float32).reshape(-1)
    if normf.size == 1:
        normf = np.asarray([float(normf[0]), 1.0], dtype=np.float32)
    elif normf.size >= 2:
        normf = normf[:2].astype(np.float32, copy=False)
    else:
        normf = np.ones(2, dtype=np.float32)

    if zseed_mode == "z0":
        zseed = np.float32(z0 + 1e-6)
    elif zseed_mode == "center":
        zseed = np.float32(Z0 / 2.0 + 1e-6)
    else:
        raise ValueError(f"Unknown zseed_mode={zseed_mode!r}; use 'z0' or 'center'.")

    if verbose:
        print(
            "Loaded global dual-channel PSF:\n"
            f"  source      : {src}\n"
            f"  ref coeff   : {coeff_ref_xyzw.shape}\n"
            f"  tar coeff   : {coeff_tar_xyzw.shape}\n"
            f"  packed coeff: {coeff_loclib_cbzyx.shape}  # (C,B,Z,Y,X)\n"
            f"  splinesize  : {splinesize.tolist()}  # [X,Y,Z,64,C]\n"
            f"  dz          : {dz}\n"
            f"  z0          : {z0}\n"
            f"  zseed       : {float(zseed):.6f}\n"
            f"  normf       : {normf.tolist()}\n"
            f"  mirror      : {mirror}"
        )

    return GlobalDualChannelSplineModel(
        coeff=coeff_loclib_cbzyx,
        splinesize=splinesize,
        zseed=zseed,
        dz=dz,
        z0=z0,
        normf=normf,
        mirror=mirror,
    )


# =============================================================================
# Likelihood normalization
# =============================================================================
def compute_llrel(
    loglikelihood: np.ndarray,
    *,
    roi_size: int,
    num_channels: int,
    em_on: bool | int = False,
) -> np.ndarray:
    """
    Compute MATLAB/SMAP-style relative log-likelihood, LLrel.

    LLrel is an exported compatibility metric used by downstream localization
    tables. It is not the raw GPU log-likelihood.
    """
    ll = np.asarray(loglikelihood, dtype=np.float32)
    emfac = np.float32(2.0 if bool(em_on) else 1.0)
    denom = np.float32((int(roi_size) ** 2) * int(num_channels))
    return np.asarray(ll * emfac / denom, dtype=np.float32)