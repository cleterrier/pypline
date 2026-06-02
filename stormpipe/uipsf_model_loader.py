# uipsf_model_loader.py

"""
uiPSF spline model loading
--------------------------
Load Python uiPSF .h5 spline PSF models and adapt them to the internal loclib
model contracts used by the pipeline.

First implementation assumptions
--------------------------------
- uiPSF has already handled any required xy-axis convention at calibration time.
- The pipeline does not perform any xy swapping.
- The selected coefficient dataset is expected at:

      /locres/<coeff_key>

  with shape:

      (C, B, Z, Y, X)

  For this dual-channel pipeline:

      C == 2
      B == 64

Internal output contracts
-------------------------
Stage 1 single-channel fitting expects per-channel coefficients:

      (B, Z, Y, X)

with splinesize:

      [X, Y, Z, B]

Stage 3 global dual-channel fitting expects coefficients:

      (C, B, Z, Y, X)

with splinesize:

      [X, Y, Z, B, C]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
from typing import Any

import numpy as np

from .mle_fitting_helpers import GlobalDualChannelSplineModel

logger = logging.getLogger("stormpipe.uipsf")


@dataclass(frozen=True)
class UiPsfSplineData:
    """
    Raw uiPSF spline data normalized to the pipeline's internal packed layout.

    coeff
        Packed dual-channel coefficient tensor, shape (2, 64, Z, Y, X).

    splinesize_global
        DLL metadata for global multichannel fitting: [X, Y, Z, 64, 2].

    dz
        Axial spline grid spacing in nm.

    z0
        Spline index corresponding to z=0 according to the pipeline convention.

    normf
        Per-channel photon normalization factors. Defaults to [1, 1].
    """
    coeff: np.ndarray
    splinesize_global: np.ndarray
    dz: float
    z0: int
    zseed: np.float32
    normf: np.ndarray
    mirror: bool
    params: dict[str, Any]
    swap_xy_axes_applied: bool

def _decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def _read_uipsf_params(h5) -> dict[str, Any]:
    """
    Read uiPSF JSON params from either a root attribute or a dataset.

    uiPSF commonly stores this as a root HDF5 attribute named "params".
    This helper is intentionally tolerant so loader failure is tied to missing
    coefficients rather than missing metadata.
    """
    raw = None

    if "params" in h5.attrs:
        raw = h5.attrs["params"]
    elif "params" in h5:
        raw = h5["params"][()]

    if raw is None:
        return {}

    try:
        return json.loads(_decode_h5_string(raw))
    except Exception:
        logger.warning("Could not parse uiPSF params JSON.", exc_info=True)
        return {}


def _pixel_size_z_to_dz_nm(params: dict[str, Any], default_nm: float = 50.0) -> float:
    """
    Extract z pixel size from uiPSF params.

    uiPSF params normally store pixel_size values in micrometers, e.g.
        pixel_size: {x: 0.097, y: 0.097, z: 0.02}

    The pipeline stores dz in nm.
    """
    pixel_size = params.get("pixel_size", {})

    try:
        z_um = float(pixel_size["z"])
        if np.isfinite(z_um) and z_um > 0:
            return float(z_um * 1000.0)
    except Exception:
        pass

    logger.warning(
        "Could not read positive params['pixel_size']['z']; using dz=%s nm.",
        default_nm,
    )
    return float(default_nm)


def _validate_uipsf_coeff_tensor(coeff: np.ndarray, *, coeff_path: str) -> tuple[int, int, int, int, int]:
    """
    Validate expected uiPSF packed coefficient layout:

        coeff.shape == (C, B, Z, Y, X)
    """
    coeff = np.asarray(coeff)

    if coeff.ndim != 5:
        raise ValueError(
            f"Expected {coeff_path} to be 5D with shape (C,64,Z,Y,X), "
            f"got {coeff.shape}"
        )

    C, B, Z, Y, X = map(int, coeff.shape)

    if C != 2:
        raise ValueError(
            f"Expected {coeff_path} channel axis C=2, got shape {coeff.shape}"
        )

    if B != 64:
        raise ValueError(
            f"Expected {coeff_path} basis axis B=64, got shape {coeff.shape}"
        )

    if Z <= 0 or Y <= 0 or X <= 0:
        raise ValueError(f"Invalid non-positive uiPSF coeff dimensions: {coeff.shape}")

    return C, B, Z, Y, X


def load_uipsf_coeff_tensor(
    h5_path: Path | str,
    *,
    coeff_key: str = "coeff",
    z0_index: int | None = None,
    normf: tuple[float, float] | np.ndarray = (1.0, 1.0),
    swap_xy_axes: bool = False,
    verbose: bool = True,
) -> UiPsfSplineData:
    """
    Load uiPSF spline coefficients in pipeline-ready packed layout.

    No xy-axis swap is performed here. uiPSF calibration is responsible for
    saving the coefficient tensor in the correct orientation.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "uiPSF .h5 loading requires h5py. Install it with `pip install h5py`."
        ) from e

    h5_path = Path(h5_path)
    coeff_path = f"locres/{coeff_key}"

    with h5py.File(str(h5_path), "r") as h5:
        params = _read_uipsf_params(h5)

        if coeff_path not in h5:
            available = []
            if "locres" in h5:
                available = sorted(str(k) for k in h5["locres"].keys())
            raise FileNotFoundError(
                f"uiPSF model does not contain dataset {coeff_path!r}. "
                f"Available /locres keys: {available}"
            )

        coeff = np.asarray(h5[coeff_path][()], dtype=np.float32)

    C, B, Z, Y, X = _validate_uipsf_coeff_tensor(
        coeff,
        coeff_path=coeff_path,
    )

    if bool(swap_xy_axes):
        logger.info(
            "Applying uiPSF xy coefficient-axis swap: %s | before=%s",
            coeff_path,
            coeff.shape,
        )
        coeff = np.transpose(coeff, (0, 1, 2, 4, 3))

    coeff = np.ascontiguousarray(coeff, dtype=np.float32)

    # Re-read dimensions after optional xy swap. For square models this shape may
    # be unchanged, but the coefficient memory/content has still been transposed.
    C, B, Z, Y, X = _validate_uipsf_coeff_tensor(
        coeff,
        coeff_path=coeff_path,
    )

    if z0_index is None:
        z0 = int(Z // 2)
    else:
        z0 = int(z0_index)
        if not (0 <= z0 < Z):
            raise ValueError(
                f"uipsf_z0_index must be in [0, {Z - 1}] for coeff shape {coeff.shape}; "
                f"got {z0}"
            )

    dz = _pixel_size_z_to_dz_nm(params)
    zseed = np.float32(z0 + 1e-6)

    normf_arr = np.asarray(normf, dtype=np.float32).reshape(-1)
    if normf_arr.shape != (2,):
        raise ValueError(f"normf must contain exactly two values, got {normf_arr.shape}")
    if not np.all(np.isfinite(normf_arr)) or np.any(normf_arr <= 0):
        raise ValueError("normf must contain two finite positive values")
    normf_arr = np.ascontiguousarray(normf_arr, dtype=np.float32)

    splinesize_global = np.asarray([X, Y, Z, B, C], dtype=np.int32)

    if verbose:
        logger.info(
            "Loaded uiPSF spline model | source=%s | dataset=%s | "
            "coeff=%s (C,B,Z,Y,X) | splinesize=%s | dz=%.6g nm | "
            "z0=%d | zseed=%.6f | normf=%s | swap_xy_axes_applied=%s",
            h5_path,
            coeff_path,
            coeff.shape,
            splinesize_global.tolist(),
            float(dz),
            int(z0),
            float(zseed),
            normf_arr.tolist(),
            bool(swap_xy_axes),
        )

    return UiPsfSplineData(
        coeff=coeff,
        splinesize_global=splinesize_global,
        dz=float(dz),
        z0=int(z0),
        zseed=zseed,
        normf=normf_arr,
        mirror=False,
        params=params,
        swap_xy_axes_applied=bool(swap_xy_axes),
    )


def load_uipsf_global_dual_channel_spline_model(
    h5_path: Path | str,
    *,
    coeff_key: str = "coeff",
    z0_index: int | None = None,
    normf: tuple[float, float] | np.ndarray = (1.0, 1.0),
    swap_xy_axes: bool = False,
    verbose: bool = True,
) -> GlobalDualChannelSplineModel:
    """
    Load a uiPSF .h5 model as the Stage 3 global dual-channel spline model.

    Returned coeff layout:
        (2, 64, Z, Y, X)

    Returned splinesize:
        [X, Y, Z, 64, 2]
    """
    data = load_uipsf_coeff_tensor(
        h5_path,
        coeff_key=coeff_key,
        z0_index=z0_index,
        normf=normf,
        swap_xy_axes=swap_xy_axes,
        verbose=verbose,
    )

    return GlobalDualChannelSplineModel(
        coeff=data.coeff,
        splinesize=data.splinesize_global,
        zseed=data.zseed,
        dz=data.dz,
        z0=data.z0,
        normf=data.normf,
        mirror=data.mirror,
    )