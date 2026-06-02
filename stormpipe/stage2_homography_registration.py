# stage2_homography_registration.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


@dataclass(frozen=True)
class RegParams:
    """
    Parameters for Stage 2 T-to-R homography estimation.
    """

    # Same-frame Euclidean mutual-nearest-neighbor pairs must be within this
    # distance in native R/T pixel coordinates.
    max_pairing_dist_px: float = 4.0

    # RANSAC reprojection threshold in pixels for homography estimation
    ransac_reproj_thresh_px: float = 2.0

    # Optional extra QC (set to None to disable)
    min_photons: float | None = None          # e.g. 300.0
    max_locprec_nm: float | None = None       # e.g. 40.0

    # Optional frame window for homography estimation.
    # End frame is exclusive. If either value is None, all available frames are used.
    frame_start: int | None = None
    frame_end: int | None = None

    # Output controls
    write_transformed_T_csv: bool = True
    transformed_T_csv_name: str = "T_fits_in_Rframe.csv"
    homography_npy_name: str = "homography_T_to_R.npy"
    homography_txt_name: str = "homography_T_to_R.txt"


def _load_channel_csv(path: Path) -> pd.DataFrame:
    """
    Load one Stage 1 localization CSV for registration.

    Required coordinates are full-frame pixel coordinates in that channel's
    native coordinate system:

        xpix, ypix

    Edge-clamped fits are retained here and removed by _apply_qc().
    """
    df = pd.read_csv(path)
    required = {"frame", "xpix", "ypix", "edge_clamped"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    # Normalize dtypes
    df["frame"] = df["frame"].astype(np.int32, copy=False)
    df["xpix"] = df["xpix"].astype(np.float32, copy=False)
    df["ypix"] = df["ypix"].astype(np.float32, copy=False)
    df["edge_clamped"] = df["edge_clamped"].astype(bool, copy=False)
    return df


def _apply_qc(df: pd.DataFrame, params: RegParams) -> pd.DataFrame:
    """
    Apply Stage 2 registration-quality filters.

    Always removes edge-clamped Stage 1 fits. Optional filters remove dim fits
    and low-precision fits when the corresponding columns are available.
    """
    mask = ~df["edge_clamped"]
    if params.min_photons is not None and "photons" in df.columns:
        mask &= (df["photons"].astype(np.float32) >= float(params.min_photons))
    if params.max_locprec_nm is not None and "locprecnm" in df.columns:
        mask &= (df["locprecnm"].astype(np.float32) <= float(params.max_locprec_nm))
    return df.loc[mask, ["frame", "xpix", "ypix"]].copy()

def _apply_frame_window(
    df: pd.DataFrame,
    *,
    frame_start: int | None,
    frame_end: int | None,
) -> pd.DataFrame:
    """
    Restrict registration localizations to an optional frame window.

    The frame_end value is exclusive. If frame_start or frame_end is None, the
    input dataframe is returned unchanged.
    """
    if frame_start is None or frame_end is None:
        return df

    f0 = int(frame_start)
    f1 = int(frame_end)

    if f0 < 0 or f1 <= f0:
        return df

    return df.loc[(df["frame"] >= f0) & (df["frame"] < f1)].copy()

def _mutual_nn_same_frame(
    R_xy: np.ndarray,
    T_xy: np.ndarray,
    max_dist_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pair same-frame R/T localizations by Euclidean mutual nearest neighbors.

    Parameters
    ----------
    R_xy
        Reflected-channel coordinates, shape (NR, 2), stored as [x, y].
    T_xy
        Transmitted-channel coordinates, shape (NT, 2), stored as [x, y].
    max_dist_px
        Maximum Euclidean nearest-neighbor distance in pixels.

    Returns
    -------
    R_matched, T_matched
        Matched coordinate arrays with shape (M, 2), both stored as [x, y].
    """
    if R_xy.size == 0 or T_xy.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    # KNN from tgt->ref and ref->tgt
    # To avoid extra deps, do a simple vectorized distance search in chunks.
    def knn(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # returns (indices, dists) of nearest b for each a
        idx = np.empty(a.shape[0], dtype=np.int32)
        dst = np.empty(a.shape[0], dtype=np.float32)
        # Chunk to limit memory
        CH = max(1, 2000 // max(1, b.shape[0]))
        for i0 in range(0, a.shape[0], CH):
            i1 = min(a.shape[0], i0 + CH)
            a_blk = a[i0:i1]  # (m,2)
            # (m,1,2) - (1,n,2) -> (m,n,2) -> (m,n)
            d2 = np.sum((a_blk[:, None, :] - b[None, :, :]) ** 2, axis=2)
            j = np.argmin(d2, axis=1)
            idx[i0:i1] = j
            dst[i0:i1] = np.sqrt(d2[np.arange(d2.shape[0]), j])
        return idx, dst

    idx_t2r, d_t2r = knn(T_xy, R_xy)
    idx_r2t, d_r2t = knn(R_xy, T_xy)

    pairs: list[tuple[int, int]] = []
    for it, ir in enumerate(idx_t2r):
        if (
            d_t2r[it] <= max_dist_px
            and idx_r2t[ir] == it
            and d_r2t[ir] <= max_dist_px
        ):
            pairs.append((ir, it))

    if not pairs:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    R_matched = np.asarray([R_xy[i] for i, _ in pairs], dtype=np.float32)
    T_matched = np.asarray([T_xy[j] for _, j in pairs], dtype=np.float32)
    return R_matched, T_matched


def _collect_pairs_by_frame(
    df_R: pd.DataFrame,
    df_T: pd.DataFrame,
    max_pairing_dist_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """
    Collect same-frame R/T point pairs for T-to-R homography estimation.

    Returns
    -------
    T_src_xy
        Transmitted-channel source points, shape (M, 2), stored as [x, y].
    R_dst_xy
        Reflected-channel destination points, shape (M, 2), stored as [x, y].
    per_frame_counts
        Mapping from frame index to number of matched pairs.
    """
    frames = np.intersect1d(df_R["frame"].unique(), df_T["frame"].unique())
    R_dst_all = []
    T_src_all = []
    per_frame_counts: dict[int, int] = {}

    for f in frames:
        R = df_R.loc[df_R["frame"] == f, ["xpix", "ypix"]].to_numpy(np.float32)
        T = df_T.loc[df_T["frame"] == f, ["xpix", "ypix"]].to_numpy(np.float32)
        if R.size == 0 or T.size == 0:
            continue
        R_matched, T_matched = _mutual_nn_same_frame(R, T, max_pairing_dist_px)
        if R_matched.size == 0:
            continue

        # H_T_to_R maps T source points into R destination points.
        R_dst_all.append(R_matched)
        T_src_all.append(T_matched)
        per_frame_counts[int(f)] = R_matched.shape[0]

    if T_src_all:
        T_src_xy = np.vstack(T_src_all)
        R_dst_xy = np.vstack(R_dst_all)
    else:
        T_src_xy = np.empty((0, 2), np.float32)
        R_dst_xy = np.empty((0, 2), np.float32)

    return T_src_xy, R_dst_xy, per_frame_counts


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def compute_projective_homography(
    outdir: Path,
    params: RegParams = RegParams(),
    *,
    csv_R: str | Path = "R_fits.csv",
    csv_T: str | Path = "T_fits.csv",
    pixelsize_nm: float | None = None,
) -> dict:
    """
    Estimate the Stage 2 projective homography from Stage 1 localization CSVs.

    The returned homography is named H_T_to_R because it maps transmitted-channel
    pixel coordinates into reflected-channel pixel coordinates:

        [x_T, y_T] -> [x_R, y_R]

    The function also saves the homography to disk and can write a diagnostic
    T_fits CSV with transformed R-frame coordinates.
    """
    outdir = Path(outdir)
    R_csv_path = Path(csv_R)
    if not R_csv_path.is_absolute():
        R_csv_path = outdir / R_csv_path

    T_csv_path = Path(csv_T)
    if not T_csv_path.is_absolute():
        T_csv_path = outdir / T_csv_path

    df_R = _load_channel_csv(R_csv_path)
    df_T = _load_channel_csv(T_csv_path)

    # Optional QC filters beyond edge-clamp
    df_R_qc_full = _apply_qc(df_R, params)
    df_T_qc_full = _apply_qc(df_T, params)

    use_window = (
        params.frame_start is not None
        and params.frame_end is not None
        and int(params.frame_start) >= 0
        and int(params.frame_end) > int(params.frame_start)
    )
    used_requested_frame_window = bool(use_window)

    if use_window:
        df_R_qc = _apply_frame_window(
            df_R_qc_full,
            frame_start=params.frame_start,
            frame_end=params.frame_end,
        )
        df_T_qc = _apply_frame_window(
            df_T_qc_full,
            frame_start=params.frame_start,
            frame_end=params.frame_end,
        )
    else:
        df_R_qc = df_R_qc_full
        df_T_qc = df_T_qc_full

    # Build frame-synchronized mutual-NN pairs within a distance cap
    T_src_xy, R_dst_xy, per_frame = _collect_pairs_by_frame(
        df_R_qc,
        df_T_qc,
        params.max_pairing_dist_px,
    )

    if use_window and T_src_xy.shape[0] < 4:
        used_requested_frame_window = False
        df_R_qc = df_R_qc_full
        df_T_qc = df_T_qc_full

        T_src_xy, R_dst_xy, per_frame = _collect_pairs_by_frame(
            df_R_qc,
            df_T_qc,
            params.max_pairing_dist_px,
        )

    if T_src_xy.shape[0] < 4:
        raise RuntimeError(
            f"Insufficient matched pairs for homography (need >=4), got {T_src_xy.shape[0]}"
        )

    # Pre-H RMSE (after nearest neighbor pairing but before any transform)
    pre_rmse_px = _rmse(T_src_xy, R_dst_xy)

    # Estimate H_T_to_R with RANSAC.
    H_T_to_R, mask = cv2.findHomography(
        T_src_xy,
        R_dst_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(params.ransac_reproj_thresh_px),
    )
    if H_T_to_R is None:
        raise RuntimeError("cv2.findHomography failed to find a model.")

    inlier_mask = (mask.ravel().astype(bool) if mask is not None else np.ones(T_src_xy.shape[0], bool))
    in_src = T_src_xy[inlier_mask]
    in_dst = R_dst_xy[inlier_mask]

    # Post-H RMSE on inliers
    # Apply H_T_to_R to T-source inliers and compare to R-destination inliers.
    in_src_h = np.hstack([in_src, np.ones((in_src.shape[0], 1), np.float32)])
    pred = (H_T_to_R @ in_src_h.T).T
    pred = pred[:, :2] / pred[:, 2:3]
    post_rmse_px = _rmse(pred, in_dst)

    # Persist homography
    np.save(outdir / params.homography_npy_name, H_T_to_R.astype(np.float64))
    np.savetxt(outdir / params.homography_txt_name, H_T_to_R.astype(np.float64), fmt="%.10f")

    # Optionally write a transformed T CSV aligned to R pixels (and nm if available)
    transformed_rows = 0
    if params.write_transformed_T_csv:
        T_all = df_T.copy()
        T_xy = T_all[["xpix", "ypix"]].to_numpy(np.float32)
        T_xy_h = np.hstack([T_xy, np.ones((T_xy.shape[0], 1), np.float32)])
        T_xy_in_R = (H_T_to_R @ T_xy_h.T).T
        T_xy_in_R = T_xy_in_R[:, :2] / T_xy_in_R[:, 2:3]
        T_all["xpix_Rframe"] = T_xy_in_R[:, 0].astype(np.float32)
        T_all["ypix_Rframe"] = T_xy_in_R[:, 1].astype(np.float32)

        if pixelsize_nm is not None:
            px_nm = float(pixelsize_nm)
            # Keep the same xnm/ynm columns but add transformed equivalents for clarity
            T_all["xnm_Rframe"] = (T_all["xpix_Rframe"] + 1.0) * px_nm
            T_all["ynm_Rframe"] = (T_all["ypix_Rframe"] + 1.0) * px_nm

        T_all.to_csv(outdir / params.transformed_T_csv_name, index=False)
        transformed_rows = int(T_all.shape[0])

    diagnostics = dict(
        total_pairs=int(T_src_xy.shape[0]),
        inliers=int(np.count_nonzero(inlier_mask)),
        inlier_ratio=float(np.mean(inlier_mask)),
        pre_rmse_px=pre_rmse_px,
        post_rmse_px=post_rmse_px,
        per_frame_pair_counts=per_frame,
        homography_path_npy=str(outdir / params.homography_npy_name),
        homography_path_txt=str(outdir / params.homography_txt_name),
        transformed_T_rows=transformed_rows,
        transformed_T_csv=(str(outdir / params.transformed_T_csv_name)
                           if params.write_transformed_T_csv else None),
        H=H_T_to_R.astype(np.float64),
        registration_used_requested_frame_window=bool(used_requested_frame_window),
        registration_frame_start=(
            int(params.frame_start) if used_requested_frame_window else None
        ),
        registration_frame_end=(
            int(params.frame_end) if used_requested_frame_window else None
        ),
    )
    return diagnostics


# Optional small CLI for quick testing
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(
        description="Compute T-to-R projective homography from Stage 1 localizations."
    )
    ap.add_argument("--outdir", required=True, type=Path, help="Folder containing R_fits.csv and T_fits.csv")
    ap.add_argument("--max_pair_px", type=float, default=4.0)
    ap.add_argument("--ransac_px", type=float, default=2.0)
    ap.add_argument("--min_photons", type=float, default=None)
    ap.add_argument("--max_locprec_nm", type=float, default=None)
    ap.add_argument("--pixelsize_nm", type=float, default=None)
    ap.add_argument("--no_write_transformed", action="store_true")
    args = ap.parse_args()

    params = RegParams(
        max_pairing_dist_px=args.max_pair_px,
        ransac_reproj_thresh_px=args.ransac_px,
        min_photons=args.min_photons,
        max_locprec_nm=args.max_locprec_nm,
        write_transformed_T_csv=not args.no_write_transformed,
    )
    diags = compute_projective_homography(
        outdir=args.outdir, params=params, pixelsize_nm=args.pixelsize_nm
    )
    print(json.dumps({k: (v if k != "H" else np.asarray(v).tolist()) for k, v in diags.items()}, indent=2))
