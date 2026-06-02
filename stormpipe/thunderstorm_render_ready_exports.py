# thunderstorm_render_ready_exports.py

"""
Render-ready localization table exports for the spectral-demixing GlobLoc pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RENDER_READY_COLUMNS = [
    "x [nm]",
    "y [nm]",
    "z [nm]",
    "frame",
    "uncertainty_xy [nm]",
    "intensity [photon]",
    "channel",
    "uncertainty_z [nm]",
    "chi2",
    "detections",
    "intensity1 [photon]",
    "intensity2 [photon]",
    "photon_ratio_T_over_total",
    "photon_ratio_T_over_R",
]


def _numeric_column(
    df: pd.DataFrame,
    column: str,
    *,
    default: float | None = None,
) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")

    if default is None:
        raise ValueError(
            f"Cannot create render-ready localization table. Missing required column: {column!r}"
        )

    return pd.Series(np.full(len(df), default), index=df.index)


def _first_available_numeric(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    default: float | None = None,
) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")

    if default is None:
        raise ValueError(
            "Cannot create render-ready localization table. Missing all candidate "
            f"columns: {columns}"
        )

    return pd.Series(np.full(len(df), default), index=df.index)


def to_render_ready_dataframe(
    locs: pd.DataFrame,
    *,
    frame_col: str = "frame",
    x_col: str = "x [nm]",
    y_col: str = "y [nm]",
    z_col: str = "z [nm]",
    channel_col: str = "channel",
    frame_offset: int = 1,
    z_sign: float = 1.0,
    z_scale: float = 1.0,
    uncertainty_xy_scale: float = 1.0,
    uncertainty_z_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Convert an internal GlobLoc localization table into the compact localization
    schema used by downstream ImageJ/ThunderSTORM rendering workflows.

    intensity1 [photon] is the reflected/reference-channel photon estimate.
    intensity2 [photon] is the transmitted/target-channel photon estimate.

    photon_ratio_T_over_R is the transmitted/reflected photon ratio. In
    fixed-ratio mode, this is the assigned fixed T/R ratio selected by the
    Stage 3 likelihood comparison and stored in the internal assigned_ratio
    column. In free-ratio mode or fallback cases, it is taken from ratio_T_over_R
    or computed as intensity2 / intensity1.

    photon_ratio_T_over_total is the transmitted fraction T / (R + T). When a
    T/R ratio is available, it is computed as:

        T/(R+T) = (T/R) / (1 + T/R)

    with a fallback to ratio_T_over_total or intensity2 / (intensity1 + intensity2).
    """
    
    if locs is None or locs.empty:
        return pd.DataFrame(columns=RENDER_READY_COLUMNS)

    df = locs.copy()

    x_nm = _numeric_column(df, x_col)
    y_nm = _numeric_column(df, y_col)
    z_nm = _numeric_column(df, z_col) * float(z_sign) * float(z_scale)

    frame = _numeric_column(df, frame_col) + int(frame_offset)

    uncertainty_xy = _first_available_numeric(
        df,
        ("locprecnm", "uncertainty_xy [nm]"),
    ) * float(uncertainty_xy_scale)

    uncertainty_z = _first_available_numeric(
        df,
        ("locprecznm", "uncertainty_z [nm]"),
    ) * float(uncertainty_z_scale)

    intensity1 = _first_available_numeric(
        df,
        ("phot1", "photons_R", "intensity1 [photon]"),
    )

    intensity2 = _first_available_numeric(
        df,
        ("phot2", "photons_T", "intensity2 [photon]"),
    )

    intensity = _first_available_numeric(
        df,
        ("phot", "intensity [photon]", "photons_R", "phot1"),
    )

    channel = _numeric_column(df, channel_col)

    chi2 = _first_available_numeric(
        df,
        ("logLikelihood", "logL", "chi2"),
    )

    detections = _first_available_numeric(
        df,
        ("numberInGroup", "detections"),
        default=1.0,
    )

    intensity1_np = intensity1.to_numpy(dtype=np.float64)
    intensity2_np = intensity2.to_numpy(dtype=np.float64)
    total_np = intensity1_np + intensity2_np

    # Prefer the fixed-ratio assignment selected during Stage 3.
    #
    # In fixed-ratio mode:
    #   assigned_ratio = selected T/R candidate ratio
    #
    # In free-ratio mode or fallback cases:
    #   use ratio_T_over_R if available, otherwise compute intensity2/intensity1.
    ratio_t_over_r = pd.Series(
        np.full(len(df), np.nan, dtype=np.float64),
        index=df.index,
        dtype=np.float64,
    )

    if "assigned_ratio" in df.columns:
        ratio_t_over_r = pd.to_numeric(
            df["assigned_ratio"],
            errors="coerce",
        ).astype(np.float64)

    if "ratio_T_over_R" in df.columns:
        ratio_t_over_r = ratio_t_over_r.fillna(
            pd.to_numeric(
                df["ratio_T_over_R"],
                errors="coerce",
            ).astype(np.float64)
        )

    computed_t_over_r = np.divide(
        intensity2_np,
        intensity1_np,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=np.isfinite(intensity1_np) & (intensity1_np > 0),
    )
    ratio_t_over_r = ratio_t_over_r.fillna(
        pd.Series(computed_t_over_r, index=df.index, dtype=np.float64)
    )

    ratio_t_over_r_np = ratio_t_over_r.to_numpy(dtype=np.float64)

    # Convert T/R to T/(R+T). This keeps fixed-ratio exports consistent with
    # settings.fixed_ratios, which are user-facing transmitted fractions.
    ratio_t_over_total_np = np.divide(
        ratio_t_over_r_np,
        1.0 + ratio_t_over_r_np,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=(
            np.isfinite(ratio_t_over_r_np)
            & np.isfinite(1.0 + ratio_t_over_r_np)
            & (np.abs(1.0 + ratio_t_over_r_np) > 1e-12)
        ),
    )

    ratio_t_over_total = pd.Series(
        ratio_t_over_total_np,
        index=df.index,
        dtype=np.float64,
    )

    if "ratio_T_over_total" in df.columns:
        ratio_t_over_total = ratio_t_over_total.fillna(
            pd.to_numeric(
                df["ratio_T_over_total"],
                errors="coerce",
            ).astype(np.float64)
        )

    computed_t_over_total = np.divide(
        intensity2_np,
        total_np,
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=np.isfinite(total_np) & (total_np > 0),
    )
    ratio_t_over_total = ratio_t_over_total.fillna(
        pd.Series(computed_t_over_total, index=df.index, dtype=np.float64)
    )

    out = pd.DataFrame(
        {
            "x [nm]": x_nm.astype(np.float64),
            "y [nm]": y_nm.astype(np.float64),
            "z [nm]": z_nm.astype(np.float32),
            "frame": frame.astype(np.int64),
            "uncertainty_xy [nm]": uncertainty_xy.astype(np.float32),
            "intensity [photon]": intensity.astype(np.float32),
            "channel": channel.astype(np.int32),
            "uncertainty_z [nm]": uncertainty_z.astype(np.float32),
            "chi2": chi2.astype(np.float32),
            "detections": detections.astype(np.int32),
            "intensity1 [photon]": intensity1.astype(np.float32),
            "intensity2 [photon]": intensity2.astype(np.float32),
            "photon_ratio_T_over_total": ratio_t_over_total.astype(np.float32),
            "photon_ratio_T_over_R": ratio_t_over_r.astype(np.float32),
        }
    )

    return out[RENDER_READY_COLUMNS]