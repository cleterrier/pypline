# stage3_filtering_grouping.py
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("stormpipe.postprocess")

def filter_localizations_stage3(
    df: pd.DataFrame,
    *,
    uncertainty_xy_threshold_nm: float,
    loglikelihood_min: float,
    max_iterations: int,
    require_converged: bool = True,
    conv_xy_threshold_px: float | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    n0 = len(out)

    # NaN rejection across all numeric columns
    if not out.empty:
        nan_mask = np.zeros(len(out), dtype=bool)
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]):
                nan_mask |= out[c].isna().to_numpy()
        out = out.loc[~nan_mask].copy()
    n1 = len(out)

    if not out.empty and "locprecnm" in out.columns:
        out = out.loc[out["locprecnm"] <= float(uncertainty_xy_threshold_nm)].copy()
    n2 = len(out)

    if not out.empty and "logLikelihood" in out.columns:
        out = out.loc[out["logLikelihood"] >= float(loglikelihood_min)].copy()
    n3 = len(out)

    if require_converged and not out.empty and "iterations" in out.columns:
        out = out.loc[out["iterations"] < int(max_iterations)].copy()
    n4 = len(out)

    if (
        not out.empty
        and conv_xy_threshold_px is not None
        and "xpix_roi_center_R" in out.columns
        and "ypix_roi_center_R" in out.columns
        and "xpix" in out.columns
        and "ypix" in out.columns
    ):
        thr = float(conv_xy_threshold_px)
        dx = np.abs(
            out["xpix"].to_numpy(dtype=np.float64)
            - out["xpix_roi_center_R"].to_numpy(dtype=np.float64)
        )
        dy = np.abs(
            out["ypix"].to_numpy(dtype=np.float64)
            - out["ypix_roi_center_R"].to_numpy(dtype=np.float64)
        )
        out = out.loc[(dx < thr) & (dy < thr)].copy()
    n5 = len(out)

    logger.debug(
        "Stage 3 filter counts | start=%d nan=%d locprec=%d LL=%d iter=%d convxy=%d",
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
    )

    return out

def filter_fixed_ratio_assignment_ambiguity(
    df: pd.DataFrame,
    *,
    ll_ratio_threshold: float = 0.999,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    n0 = len(out)

    required = {"fixed_ratio_ll_best", "fixed_ratio_ll_second"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(
            "filter_fixed_ratio_assignment_ambiguity requires columns: "
            f"{sorted(required)}; missing {sorted(missing)}"
        )

    best = pd.to_numeric(out["fixed_ratio_ll_best"], errors="coerce").to_numpy(dtype=np.float64)
    second = pd.to_numeric(out["fixed_ratio_ll_second"], errors="coerce").to_numpy(dtype=np.float64)

    ll_ratio = np.full(len(out), np.nan, dtype=np.float64)
    valid = np.isfinite(best) & np.isfinite(second) & (second != 0.0)
    ll_ratio[valid] = best[valid] / second[valid]

    out["fixed_ratio_ll_ratio"] = ll_ratio.astype(np.float32)

    # If second-best LL is unavailable (for example only one tested ratio),
    # keep the row; otherwise apply the ambiguity filter.
    keep = (~valid) | (ll_ratio < float(ll_ratio_threshold))
    out = out.loc[keep].copy()
    n1 = len(out)

    logger.debug(
        "Fixed-ratio ambiguity filter | start=%d kept=%d dropped=%d threshold=%g",
        n0,
        n1,
        n0 - n1,
        float(ll_ratio_threshold),
    )

    return out

def _assign_temporal_groups_single(
    df: pd.DataFrame,
    *,
    group_dx_px: float = 1.0,
    group_dt_frames: int = 1,
) -> pd.DataFrame:
    """
    Assign temporal-spatial groups (i.e. blinks that span several frames) with a forward tracker.

      - localizations are sorted by frame, then x
      - each unassigned localization starts a new group
      - groups extend forward in time
      - in each next frame, the first unassigned localization satisfying
            |x - xh| < group_dx_px  and  |y - yh| < group_dx_px
        is assigned to the same group
      - after each match, the running track center is updated as:
            xh = (xh + x_new) / 2
            yh = (yh + y_new) / 2
      - if no match is found in the next frame, a dark-frame counter is
        incremented; tracking stops once numdark > group_dt_frames

    Returns
    -------
    DataFrame
        Input dataframe plus:
          - groupindex
          - numberInGroup
    """
    if df is None or df.empty:
        out = df.copy()
        if "groupindex" not in out.columns:
            out["groupindex"] = pd.Series(dtype=np.int64)
        if "numberInGroup" not in out.columns:
            out["numberInGroup"] = pd.Series(dtype=np.int32)
        return out

    required = {"frame", "xpix", "ypix"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"assign_temporal_groups missing columns: {sorted(missing)}")

    out = df.copy().reset_index(drop=True)
    n = len(out)

    frames = out["frame"].to_numpy(dtype=np.int64)
    x = out["xpix"].to_numpy(dtype=np.float64)
    y = out["ypix"].to_numpy(dtype=np.float64)

    # Sort by frame, then x
    order = np.lexsort((x, frames))
    frames_s = frames[order]
    x_s = x[order]
    y_s = y[order]

    dX = float(group_dx_px)
    dT = int(group_dt_frames)

    group_sorted = np.zeros(n, dtype=np.int64)
    group_id = 0

    # frame -> [start, end) index range in sorted arrays
    unique_frames, frame_starts = np.unique(frames_s, return_index=True)
    frame_ends = np.r_[frame_starts[1:], n]
    frame_to_range = {
        int(f): (int(s), int(e))
        for f, s, e in zip(unique_frames, frame_starts, frame_ends)
    }

    thisentry = 0
    while thisentry < n:
        # find next unassigned localization
        while thisentry < n and group_sorted[thisentry] > 0:
            thisentry += 1
        if thisentry >= n:
            break

        group_id += 1

        xh = x_s[thisentry]
        yh = y_s[thisentry]
        frh = int(frames_s[thisentry])
        group_sorted[thisentry] = group_id

        numdark = 0

        while numdark <= dT:
            frtest = frh + 1
            rng = frame_to_range.get(frtest, None)

            particlefound = False
            if rng is not None:
                start, end = rng

                # MATLAB code exploits x-sorting and only scans the next frame
                j = start
                while j < end and x_s[j] < xh - dX:
                    j += 1

                while j < end and x_s[j] < xh + dX:
                    if (
                        group_sorted[j] == 0
                        and (y_s[j] > yh - dX)
                        and (y_s[j] < yh + dX)
                    ):
                        group_sorted[j] = group_id
                        xh = (xh + x_s[j]) / 2.0
                        yh = (yh + y_s[j]) / 2.0
                        frh = int(frames_s[j])
                        numdark = 0
                        particlefound = True
                        break
                    j += 1

            if not particlefound:
                frh = frtest
                numdark += 1

    # count group sizes in sorted order
    counts = np.bincount(group_sorted)
    number_in_group_sorted = counts[group_sorted].astype(np.int32)

    # unsort back to original row order
    groupindex = np.empty(n, dtype=np.int64)
    number_in_group = np.empty(n, dtype=np.int32)
    groupindex[order] = group_sorted
    number_in_group[order] = number_in_group_sorted

    out["groupindex"] = groupindex
    out["numberInGroup"] = number_in_group
    return out

def assign_temporal_groups(
    df: pd.DataFrame,
    *,
    group_dx_px: float = 1.0,
    group_dt_frames: int = 1,
) -> pd.DataFrame:
    """
    Assign temporal-spatial groups (i.e. blinks that span several frames) with a forward tracker.

    If a 'channel' column is present, grouping is performed independently
    within each channel so localizations from different assigned channels
    can never be merged into the same group.
    """
    if df is None or df.empty:
        out = df.copy()
        if "groupindex" not in out.columns:
            out["groupindex"] = pd.Series(dtype=np.int64)
        if "numberInGroup" not in out.columns:
            out["numberInGroup"] = pd.Series(dtype=np.int32)
        return out

    required = {"frame", "xpix", "ypix"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"assign_temporal_groups missing columns: {sorted(missing)}")

    # No channel column -> keep existing behavior
    if "channel" not in df.columns:
        return _assign_temporal_groups_single(
            df,
            group_dx_px=group_dx_px,
            group_dt_frames=group_dt_frames,
        )

    out = df.copy().reset_index(drop=True)

    groupindex = np.zeros(len(out), dtype=np.int64)
    number_in_group = np.zeros(len(out), dtype=np.int32)

    next_gid = 1

    # Group each assigned channel independently
    channel_vals = np.sort(pd.unique(out["channel"]))
    for ch in channel_vals:
        mask = (out["channel"] == ch).to_numpy()
        if not np.any(mask):
            continue

        sub = out.loc[mask].copy().reset_index()
        grouped = _assign_temporal_groups_single(
            sub,
            group_dx_px=group_dx_px,
            group_dt_frames=group_dt_frames,
        )

        sub_idx = grouped["index"].to_numpy(dtype=np.int64)
        sub_gid = grouped["groupindex"].to_numpy(dtype=np.int64)

        if sub_gid.size == 0:
            continue

        # Offset so group IDs remain unique across channels
        sub_gid = sub_gid + (next_gid - 1)

        groupindex[sub_idx] = sub_gid
        number_in_group[sub_idx] = grouped["numberInGroup"].to_numpy(dtype=np.int32)

        next_gid = int(sub_gid.max()) + 1

    out["groupindex"] = groupindex
    out["numberInGroup"] = number_in_group
    return out

def filter_groups_by_size(
    df: pd.DataFrame,
    *,
    max_number_in_group: int = 5,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()
    if "numberInGroup" not in df.columns:
        return df.copy()
    return df.loc[df["numberInGroup"] <= int(max_number_in_group)].copy()


def combine_grouped_localizations(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()

    if "groupindex" not in df.columns:
        return df.copy()

    out_rows: list[dict] = []

    for gid, g in df.groupby("groupindex", sort=True):
        row: dict = {}

        # MATLAB-like group weights: 1 / locprecnm^2 when available
        if "locprecnm" in g.columns:
            lp = pd.to_numeric(g["locprecnm"], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(lp) & (lp > 0)
            weights = np.zeros(len(g), dtype=np.float64)
            weights[valid] = 1.0 / (lp[valid] * lp[valid])
        else:
            weights = np.ones(len(g), dtype=np.float64)

        def weighted_mean(series: pd.Series) -> float:
            vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            m = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
            if not np.any(m):
                return np.nan
            return float(np.sum(vals[m] * weights[m]) / np.sum(weights[m]))

        def summed(series: pd.Series) -> float:
            vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(np.sum(vals)) if vals.size > 0 else np.nan

        def invvar_combine(series: pd.Series) -> float:
            vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size == 0:
                return np.nan
            return float(1.0 / np.sqrt(np.sum(1.0 / (vals * vals))))

        def first_valid(series: pd.Series):
            nonnull = series.dropna()
            if len(nonnull) == 0:
                return np.nan
            return nonnull.iloc[0]

        def first_valid_numeric(series: pd.Series) -> float:
            vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(vals[0]) if vals.size > 0 else np.nan

        # Core metadata
        if "pair_id" in g.columns:
            row["pair_id"] = int(first_valid_numeric(g["pair_id"]))
        if "frame" in g.columns:
            row["frame"] = int(pd.to_numeric(g["frame"], errors="coerce").min())
        row["groupindex"] = int(gid)
        row["numberInGroup"] = int(len(g))

        # Channel / assignment metadata
        passthrough_first_cols = [
            "channel",
            "assigned_ratio",
            "fixed_ratio_ll_best",
            "fixed_ratio_ll_second",
            "fixed_ratio_ll_margin",
            "fixed_ratio_ll_ratio",
        ]
        for c in passthrough_first_cols:
            if c in g.columns:
                if pd.api.types.is_numeric_dtype(g[c]):
                    row[c] = first_valid_numeric(g[c])
                else:
                    row[c] = first_valid(g[c])

        # Use max for these per-group summary fields
        max_cols = ["logLikelihood", "LLrel", "iterations"]
        for c in max_cols:
            if c in g.columns:
                vals = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                row[c] = float(np.max(vals)) if vals.size > 0 else np.nan

        # Weighted mean columns
        mean_cols = [
            "x [nm]", "y [nm]", "z [nm]",
            "xpix", "ypix",
            "xpix_roi_center_R", "ypix_roi_center_R",
            "xpix_roi_center_T", "ypix_roi_center_T",
        ]
        for c in mean_cols:
            if c in g.columns:
                row[c] = weighted_mean(g[c])

        # Sum columns
        sum_cols = [
            "bg1", "bg2",
            "phot", "phot1", "phot2",
            "photons_total",
        ]
        for c in sum_cols:
            if c in g.columns:
                row[c] = summed(g[c])

        # Recompute ratio columns from grouped photometry
        if "phot1" in row and "phot2" in row and np.isfinite(row["phot1"]) and np.isfinite(row["phot2"]):
            p1 = float(row["phot1"])
            p2 = float(row["phot2"])
            row["ratio_T_over_R"] = (p2 / p1) if p1 > 0 else np.nan
            row["ratio_T_over_total"] = (p2 / (p1 + p2)) if (p1 + p2) > 0 else np.nan

        # Inverse-variance combination for uncertainty/error-like columns
        err_cols = [
            "locprecnm", "locprecznm",
            "xpixerr", "ypixerr",
            "xnmerr", "ynmerr", "znmerr",
            "bg1err", "bg2err",
            "photerr", "phot1err", "phot2err",
        ]
        for c in err_cols:
            if c in g.columns:
                row[c] = invvar_combine(g[c])

        # Generic fallback: preserve any remaining columns
        handled = set(row.keys())
        for c in g.columns:
            if c in handled:
                continue

            s = g[c]

            if pd.api.types.is_numeric_dtype(s):
                vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
                finite = vals[np.isfinite(vals)]
                if finite.size == 0:
                    row[c] = np.nan
                elif np.allclose(finite, finite[0], rtol=0.0, atol=0.0):
                    row[c] = float(finite[0])
                else:
                    row[c] = float(finite[0])
            else:
                row[c] = first_valid(s)

        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    preferred = [
        "pair_id", "frame",
        "x [nm]", "y [nm]", "z [nm]",
        "locprecnm", "locprecznm",
        "channel", "assigned_ratio",
        "bg1", "bg1err", "bg2", "bg2err",
        "iterations", "logLikelihood", "LLrel",
        "fixed_ratio_ll_best", "fixed_ratio_ll_second",
        "fixed_ratio_ll_margin", "fixed_ratio_ll_ratio",
        "phot", "photerr",
        "phot1", "phot1err", "phot2", "phot2err",
        "photons_total",
        "ratio_T_over_R", "ratio_T_over_total",
        "xpix", "ypix", "xpixerr", "ypixerr",
        "xnmerr", "ynmerr", "znmerr",
        "groupindex", "numberInGroup",
        "xpix_roi_center_R", "ypix_roi_center_R",
        "xpix_roi_center_T", "ypix_roi_center_T",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]

def filter_stage3_dataframe(
    df: pd.DataFrame,
    settings,
) -> pd.DataFrame:
    """
    Apply the standard Stage 3 localization filters using pipeline settings.
    """
    return filter_localizations_stage3(
        df,
        uncertainty_xy_threshold_nm=float(settings.stage3_uncertainty_xy_threshold_nm),
        loglikelihood_min=-(float(settings.roi_size) ** 2),
        max_iterations=int(settings.gpu_iterations),
        require_converged=bool(settings.stage3_require_converged),
        conv_xy_threshold_px=(
            float(settings.stage3_conv_xy_threshold_px)
            if getattr(settings, "stage3_convergence_xy_filter", True)
            else None
        ),
    )


def group_stage3_dataframe(
    df: pd.DataFrame,
    settings,
) -> pd.DataFrame:
    """
    Apply Stage 3 temporal grouping metadata.

    This function deliberately does NOT combine grouped localizations. Group
    combination is reserved for final render-ready channel exports after drift
    correction has been applied.
    """
    if df is None or df.empty:
        return df.copy()

    out = df.copy()

    if getattr(settings, "stage3_grouping_enabled", True):
        out = assign_temporal_groups(
            out,
            group_dx_px=float(settings.stage3_group_dx_px),
            group_dt_frames=int(settings.stage3_group_dt_frames),
        )
        logger.info("After group assignment: %d", len(out))
        logger.info("Keeping individual localization rows; group metadata only.")

    return out