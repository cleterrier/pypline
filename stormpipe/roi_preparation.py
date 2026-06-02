# roi_preparation.py

"""
Preprocessing pipeline
----------------------
CPU-side detection, ROI extraction, and Stage 3 paired-ROI preparation for the
ratiometric STORM pipeline.

Responsibilities
----------------
Stage 1 preprocessing:
    Detect candidate peaks independently per channel, cut image-order ROIs around
    integer seed pixels, and stream those ROIs to the Stage 1 GPU fitter.

Stage 3 preprocessing:
    Load Stage 1 fitted localizations, transform transmitted-channel coordinates
    into reflected-channel space, conservatively pair R/T detections, merge paired
    coordinates, preserve per-channel residuals, and cut paired image-order ROIs
    for global fitting.

Array contracts
---------------
- Images are indexed as image[y, x].
- Peak centers and ROI centers are stored as [x, y] full-frame pixel coordinates.
- ROI cutouts produced here are image-order arrays with shape (N, Y, X).
- GPU helpers later convert image-order ROIs to loclib CUDA layout.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Rectangle
from numpy.lib.stride_tricks import sliding_window_view

from .preprocessing_utils import (
    get_dynamic_cutoff,
    apply_homography,
    pair_mutual_linf_one_to_one,
    centers_and_residuals,
    cut_rois_at_centers,
    count_readable_tiff_prefix,
    safe_read_tiff_frame,
    adu_to_photons,
)

__all__ = [
    "noise_correct",
    "peak_detect",
    "extract_rois",
    "detect_peaks_in_frame",
    "gather_detections",
    "display_preview_detections",
    "iter_single_channel_roi_batches_from_tiff",
    "iter_stage3_paired_roi_batches_from_stage1_csvs",
    "combine_and_cut_rois_stage3_from_stage1_one_frame",
]

# =========================
# 1) Denoise (Anscombe-like transform and baseline shift)
# =========================
def noise_correct(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32, copy=False)
    transformed = 2.0 * np.sqrt(np.clip(img, 0.0, None) + 0.375)
    corrected = transformed - float(transformed.min())
    return corrected.astype(np.float32, copy=False)


# =========================
# 2) Peak detection (DoG + dynamic cutoff + non-maximum suppression)
# =========================
def peak_detect(
    frame: np.ndarray,
    *,
    threshold_factor: float = 1.0,
    sigma1: float = 1.2,
    sigma2: float = 3.0,
    keep_peak_winner: bool = True,
    min_distance: int = 7,
) -> list[tuple[int, int, float]]:
    """
    Detect candidate peaks using DoG + dynamic threshold + 3x3 local maxima.

    Modes
    -----
    keep_peak_winner=True
        Non-maximum suppression using an L∞ / square neighborhood:
        among candidate peaks within `min_distance` pixels, keep only the
        strongest peak.

    keep_peak_winner=False
        Reject clustered detections: keep only candidate peaks that have no
        other candidate peak within `min_distance` pixels.

    Returns
    -------
    list[(y:int, x:int, intensity:float)]
        Integer pixel centers in image indexing order. The GPU fitter later
        refines these to subpixel coordinates.
    """
    import cv2
    from scipy.ndimage import maximum_filter
    from scipy.spatial import cKDTree

    img = frame.astype(np.float32, copy=False)

    # --- DoG (separable, multithreaded) ---
    s1 = max(1e-6, float(sigma1))
    s2 = max(1e-6, float(sigma2))
    g1 = cv2.GaussianBlur(img, (0, 0), s1, borderType=cv2.BORDER_REFLECT_101)
    g2 = cv2.GaussianBlur(img, (0, 0), s2, borderType=cv2.BORDER_REFLECT_101)
    dog = (g1 - g2).astype(np.float32, copy=False)

    # --- Dynamic threshold + 3x3 local maxima (candidates) ---
    cutoff = get_dynamic_cutoff(dog, float(threshold_factor))
    neigh3 = maximum_filter(dog, size=3, mode="nearest")
    cand_mask = (dog == neigh3) & (dog > cutoff)
    if not np.any(cand_mask):
        return []

    Y, X = dog.shape
    r = int(max(0, min_distance))
    yx = np.argwhere(cand_mask)          # (Nc, 2) [y,x]
    vals = dog[cand_mask].astype(np.float32, copy=False)
    Nc = yx.shape[0]

    def _pack(coords: np.ndarray, intens: np.ndarray) -> list[tuple[int, int, float]]:
        if coords.size == 0:
            return []
        return [(int(y), int(x), float(I)) for (y, x), I in zip(coords, intens)]

    # === Mode A: keep strongest per L∞-neighborhood (classic NMS) ===
    if keep_peak_winner or r <= 0:
        if r <= 0:
            return _pack(yx, vals)
        order = np.argsort(vals)[::-1]       # strong → weak
        yx_sorted = yx[order]
        vals_sorted = vals[order]
        tree = cKDTree(yx_sorted)
        kept = np.zeros(Nc, dtype=bool)
        suppressed = np.zeros(Nc, dtype=bool)
        for i in range(Nc):
            if suppressed[i]:
                continue
            kept[i] = True
            nbrs = tree.query_ball_point(yx_sorted[i], r=r, p=np.inf)  # square radius
            suppressed[nbrs] = True
        coords_keep = yx_sorted[kept]
        intens_keep = vals_sorted[kept]
        return _pack(coords_keep, intens_keep)

    # === Mode B: reject any peak that has another peak within min_distance ===
    # Integral image approach for fast local counts.
    m = cand_mask.astype(np.int32, copy=False)
    II = np.zeros((Y + 1, X + 1), dtype=np.int32)
    np.cumsum(np.cumsum(m, axis=0), axis=1, out=II[1:, 1:])

    y = yx[:, 0]
    x = yx[:, 1]
    y0 = np.maximum(0, y - r)
    y1 = np.minimum(Y - 1, y + r)
    x0 = np.maximum(0, x - r)
    x1 = np.minimum(X - 1, x + r)

    counts = (
        II[y1 + 1, x1 + 1]
        - II[y0, x1 + 1]
        - II[y1 + 1, x0]
        + II[y0, x0]
    )

    singles_mask = counts == 1
    coords_keep = yx[singles_mask]
    intens_keep = vals[singles_mask]

    return _pack(coords_keep, intens_keep)


# =========================
# 3) Extract square ROIs (vectorized)
# =========================
def extract_rois(
    frame: np.ndarray,
    peaks: list[tuple[int, int, float]],
    *,
    roi_size: int = 13,
    frame_index: int = 0,
) -> list[dict]:
    """
    Cut image-order ROIs around integer peak detections.

    Parameters
    ----------
    frame
        Photon-corrected image with shape (Y, X).
    peaks
        Peak detections as (y, x, intensity). Detection uses image indexing, but
        exported seed coordinates are stored as xpix/ypix.
    roi_size
        Odd ROI side length.
    frame_index
        Frame index assigned to each ROI metadata record.

    Returns
    -------
    list[dict]
        Each dict contains:
            roi       : image-order ROI, shape (K, K), equivalent to (Y, X)
            xpix      : integer x seed in full-frame pixels
            ypix      : integer y seed in full-frame pixels
            frame     : frame index
            offset_x  : same as xpix, reserved for future variants
            offset_y  : same as ypix, reserved for future variants
            intensity : integrated ROI intensity on the photon-corrected image
    """
    if roi_size % 2 == 0:
        raise ValueError(f"roi_size must be odd, got {roi_size}")

    half = (roi_size - 1) // 2
    Y, X = frame.shape
    if len(peaks) == 0:
        return []

    coords = np.array([(int(y), int(x)) for y, x, _ in peaks], np.int32)
    valid = (
        (coords[:, 0] >= half) & (coords[:, 0] < Y - half) &
        (coords[:, 1] >= half) & (coords[:, 1] < X - half)
    )
    coords = coords[valid]
    if coords.size == 0:
        return []

    win = sliding_window_view(frame, (roi_size, roi_size))   # (Y-K+1, X-K+1, K, K)
    rois_view = win[coords[:, 0] - half, coords[:, 1] - half]  # (N,K,K)

    # Vectorized integrated intensity per ROI (on photon-corrected image)
    sums = rois_view.reshape(rois_view.shape[0], -1).sum(axis=1, dtype=np.float32)

    out: list[dict] = []
    for (y, x), roi, s in zip(coords, rois_view, sums):
        out.append({
            "roi": np.ascontiguousarray(roi, dtype=np.float32),
            "xpix": int(x),
            "ypix": int(y),
            "frame": int(frame_index),
            "offset_x": int(x),
            "offset_y": int(y),
            "intensity": float(s),
        })
    return out


def detect_peaks_in_frame(
    frame: np.ndarray,
    *,
    thr: float,
    s1: float,
    s2: float,
    keep_peak_winner: bool,
    mindist: int,
):
    filtered = noise_correct(frame)
    return peak_detect(
        filtered,
        threshold_factor=thr,
        sigma1=s1,
        sigma2=s2,
        keep_peak_winner=keep_peak_winner,
        min_distance=mindist,
    )


def gather_detections(
    stack: np.ndarray,
    *,
    roi_size: int,
    thr: float,
    s1: float,
    s2: float,
    keep_peak_winner: bool,
    mindist: int,
    max_frames: int | None = None,
):
    n_total = int(stack.shape[0])
    n_frames = n_total if max_frames is None else min(int(max_frames), n_total)

    detections = []
    rois_meta = []
    for f in range(n_frames):
        frame = stack[f]
        peaks = detect_peaks_in_frame(
            frame,
            thr=thr,
            s1=s1,
            s2=s2,
            keep_peak_winner=keep_peak_winner,
            mindist=mindist,
        )
        # Cut ROIs first (this excludes near-border peaks automatically)
        frame_rois = extract_rois(frame, peaks, roi_size=roi_size, frame_index=f)
        rois_meta.extend(frame_rois)

        # Use integrated ROI intensity (on photon-corrected frame) for detections
        for r in frame_rois:
            detections.append({
                "frame": f,
                "xpix": int(r["xpix"]),
                "ypix": int(r["ypix"]),
                "intensity": float(r["intensity"]),
            })

    return detections, rois_meta


def display_preview_detections(
    stack_R: np.ndarray,
    stack_T: np.ndarray,
    frame_index: int,
    *,
    roi_size: int,
    thr_R,
    thr_T,
    s1: float,
    s2: float,
    keep_peak_winner: bool,
    mindist: int,
    display_index: int | None = None,
):
    """
    Preview using the same ROI-generation path:
    Build 1-frame stacks and run gather_detections(), then draw ROI boxes.
    """
    if stack_R.ndim != 3 or stack_T.ndim != 3:
        raise ValueError("Preview expects image-order stacks with shape (F, Y, X).")

    nR, nT = stack_R.shape[0], stack_T.shape[0]
    if nR == 0 or nT == 0:
        raise RuntimeError("Empty stacks for R or T.")

    f = max(0, min(frame_index, min(nR, nT) - 1))
    one_R = stack_R[f:f+1]
    one_T = stack_T[f:f+1]

    _, rois_R = gather_detections(
        one_R,
        roi_size=roi_size,
        thr=thr_R,
        s1=s1,
        s2=s2,
        keep_peak_winner=keep_peak_winner,
        mindist=mindist,
        max_frames=1,
    )
    _, rois_T = gather_detections(
        one_T,
        roi_size=roi_size,
        thr=thr_T,
        s1=s1,
        s2=s2,
        keep_peak_winner=keep_peak_winner,
        mindist=mindist,
        max_frames=1,
    )

    display_f = f if display_index is None else int(display_index)
    half = roi_size // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fr = one_R[0]; ft = one_T[0]

    for ax, img, rois_meta, title in [
        (axes[0], fr, rois_R, "Reflected (ch0)"),
        (axes[1], ft, rois_T, "Transmitted (ch1)"),
    ]:
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(f"{title}: frame {display_f} • {len(rois_meta)} ROIs • roi={roi_size}")
        ax.axis("off")
        for m in rois_meta:
            x = m["xpix"]; y = m["ypix"]
            x0_edge = (x - half) - 0.5
            y0_edge = (y - half) - 0.5
            rect = Rectangle((x0_edge, y0_edge), roi_size, roi_size, fill=False, linewidth=0.8, edgecolor="r")
            ax.add_patch(rect)

    return len(rois_R), len(rois_T), display_f


def iter_single_channel_roi_batches_from_tiff(
    tif_path: Path | str,
    *,
    offset: float,
    camera_gain: float,
    QE: float,
    roi_size: int,
    thr: float,
    s1: float,
    s2: float,
    keep_peak_winner: bool,
    mindist: int,
    batch_size: int,
    max_frames: int | None = None,
    start_frame: int = 0,
    roi_id_start: int = 0,
    on_frame=None,
):
    """
    Yield GPU-sized batches of single-channel ROI cutouts from one TIFF stack.
    (Pipeline stage 1)

    This is an offline file-backed iterator. It reads frames incrementally, photon-
    corrects each frame, detects candidate peaks, cuts ROIs, buffers them until
    batch_size is reached, and yields one ROI batch plus metadata at a time.

    Steps per frame:
        1) photon-correct: max((adu raw - offset) * gain * QE)
           **clip negative values so hit floor at zero!
        2) filter image and detect candidate peaks
        3) cut image-order ROIs from the photon-corrected image
        4) accumulate ROIs until batch_size is reached
        5) yield (rois, xpix, ypix, frame, ids)

    Yields
    ------
    rois
        Image-order ROI cutouts with shape (N, Y, X), float32 photons.
    xpix, ypix
        Integer seed centers in full-frame pixel coordinates, shape (N,).
    frame
        Frame index in TIFF stack, shape (N,).
    ids
        Unique ROI IDs, shape (N,).

    Notes
    -----
    GPU helper functions convert these image-order ROIs to loclib CUDA layout before
    calling the single-channel fitter.
    """
    if roi_size % 2 == 0:
        raise ValueError("roi_size must be odd")

    roi_id = int(roi_id_start)

    buf_rois: list[np.ndarray] = []
    buf_x: list[int] = []
    buf_y: list[int] = []
    buf_f: list[int] = []
    buf_id: list[int] = []

    _reported, readable = count_readable_tiff_prefix(tif_path)
    total = readable

    with tifffile.TiffFile(str(tif_path)) as tif:
        f_end = min(total, start_frame + (max_frames if max_frames is not None else total))

        for f in range(start_frame, f_end):
            # 1) photon-correct the frame
            frame = safe_read_tiff_frame(tif, f, tif_path=tif_path)
            if frame is None:
                if on_frame is not None:
                    try:
                        on_frame(f)
                    except Exception:
                        pass
                continue

            photons = adu_to_photons(
                frame,
                offset=offset,
                camera_gain=camera_gain,
                QE=QE,
            )

            # 2) detect peaks
            peaks = detect_peaks_in_frame(
                photons,
                thr=thr,
                s1=s1,
                s2=s2,
                keep_peak_winner=keep_peak_winner,
                mindist=mindist,
            )

            # 3) cut ROIs and push into buffers
            rois_meta = extract_rois(photons, peaks, roi_size=roi_size, frame_index=f)
            for r in rois_meta:
                buf_rois.append(r["roi"].astype(np.float32, copy=False))
                buf_x.append(int(r["xpix"]))
                buf_y.append(int(r["ypix"]))
                buf_f.append(int(r["frame"]))
                buf_id.append(roi_id)
                roi_id += 1

            if on_frame is not None:
                try:
                    on_frame(f)
                except Exception:
                    pass

            # 4) flush if batch is full
            if len(buf_rois) >= batch_size:
                rois = np.stack(buf_rois, axis=0) if buf_rois else np.empty((0, roi_size, roi_size), np.float32)
                xpix = np.asarray(buf_x, dtype=np.int32)
                ypix = np.asarray(buf_y, dtype=np.int32)
                frameN = np.asarray(buf_f, dtype=np.int32)
                ids = np.asarray(buf_id, dtype=np.int64)
                yield rois, xpix, ypix, frameN, ids
                buf_rois.clear(); buf_x.clear(); buf_y.clear(); buf_f.clear(); buf_id.clear()

    # final flush
    if buf_rois:
        rois = np.stack(buf_rois, axis=0)
        xpix = np.asarray(buf_x, dtype=np.int32)
        ypix = np.asarray(buf_y, dtype=np.int32)
        frameN = np.asarray(buf_f, dtype=np.int32)
        ids = np.asarray(buf_id, dtype=np.int64)
        yield rois, xpix, ypix, frameN, ids

def _load_stage1_locs_csv(
    csv_path: Path | str,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """
    Load Stage 1 localization CSV rows needed for Stage 3 pairing.

    The returned dataframe contains numeric frame, xpix, ypix, zidx, locprecnm, and
    locprecznm columns, optionally restricted to a frame window. Coordinates are
    full-frame pixel coordinates in that channel's native coordinate space.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage 1 localization CSV not found: {csv_path}")

    usecols = [
        "frame",
        "xpix",
        "ypix",
        "zidx",
        "locprecnm",
        "locprecznm",
    ]
    df = pd.read_csv(csv_path, usecols=usecols)

    f0 = int(max(0, start_frame))
    if max_frames is None:
        mask = df["frame"] >= f0
    else:
        fend = f0 + int(max_frames)
        mask = (df["frame"] >= f0) & (df["frame"] < fend)

    df = df.loc[mask].copy()
    if df.empty:
        return df

    for c in usecols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["frame", "xpix", "ypix", "zidx"]).copy()
    df["frame"] = df["frame"].astype(np.int32)

    return df.sort_values(["frame", "xpix", "ypix"]).reset_index(drop=True)


def _group_stage1_locs_by_frame(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    return {int(f): g.copy() for f, g in df.groupby("frame", sort=False)}


def combine_and_cut_rois_stage3_from_stage1_one_frame(
    frame_idx: int,
    R_phot: np.ndarray,
    T_phot: np.ndarray,
    R_locs: pd.DataFrame,
    T_locs: pd.DataFrame,
    H_T_to_R: np.ndarray,
    *,
    roi_size: int,
    pair_max_linf_dist_px: float = 5.0,
) -> dict:
    """
    Build paired Stage 3 ROIs for one frame using Stage 1 fitted localizations.

    Inputs
    ------
    R_phot, T_phot
        Photon-corrected images with shape (Y, X).
    R_locs, T_locs
        Stage 1 localization rows for this frame. xpix/ypix are full-frame pixel
        coordinates in each channel's native coordinate system.
    H_T_to_R
        Homography mapping transmitted-channel pixel coordinates into reflected-
        channel coordinate space.

    Processing steps
    ----------------
    1. Read Stage 1 fitted xpix/ypix seeds.
    2. Transform T coordinates into R space.
    3. Pair R and transformed-T coordinates using conservative mutual L∞ matching.
    4. Merge paired xy coordinates in R space using inverse-variance weights from
       locprecnm.
    5. Merge zidx using inverse-variance weights from locprecznm.
    6. Round merged coordinates to per-channel integer ROI centers.
    7. Preserve per-channel subpixel residuals as [dx, dy].
    8. Cut paired image-order R/T ROIs.
    """

    assert roi_size % 2 == 1, "roi_size must be odd"

    if R_locs is None or T_locs is None or R_locs.empty or T_locs.empty:
        return dict(
            frame=frame_idx,
            n_pairs=0,
            rois_R=np.empty((0, roi_size, roi_size), np.float32),
            rois_T=np.empty((0, roi_size, roi_size), np.float32),
            zseed_abs=np.empty((0,), np.float32),
        )

    R_xy = R_locs[["xpix", "ypix"]].to_numpy(dtype=np.float32, copy=True)
    T_xy = T_locs[["xpix", "ypix"]].to_numpy(dtype=np.float32, copy=True)

    T_in_R = apply_homography(T_xy, H_T_to_R)

    idxR, idxT = pair_mutual_linf_one_to_one(
        R_xy.astype(np.float32),
        T_in_R.astype(np.float32),
        max_linf_dist=float(pair_max_linf_dist_px),
    )
    if idxR.size == 0:
        return dict(
            frame=frame_idx,
            n_pairs=0,
            rois_R=np.empty((0, roi_size, roi_size), np.float32),
            rois_T=np.empty((0, roi_size, roi_size), np.float32),
            zseed_abs=np.empty((0,), np.float32),
        )

    # XY weights: 1 / locprecnm^2
    R_xy_prec = pd.to_numeric(R_locs["locprecnm"], errors="coerce").to_numpy(dtype=np.float32)
    T_xy_prec = pd.to_numeric(T_locs["locprecnm"], errors="coerce").to_numpy(dtype=np.float32)

    wR_xy = 1.0 / np.maximum(R_xy_prec[idxR], 1e-6) ** 2
    wT_xy = 1.0 / np.maximum(T_xy_prec[idxT], 1e-6) ** 2
    wsum_xy = wR_xy + wT_xy
    wsum_xy[wsum_xy == 0.0] = 1.0

    merged_R_x = (wR_xy * R_xy[idxR, 0] + wT_xy * T_in_R[idxT, 0]) / wsum_xy
    merged_R_y = (wR_xy * R_xy[idxR, 1] + wT_xy * T_in_R[idxT, 1]) / wsum_xy
    merged_R = np.stack([merged_R_x, merged_R_y], axis=1).astype(np.float32)

    # Z weights: 1 / locprecznm^2
    R_zidx = pd.to_numeric(R_locs["zidx"], errors="coerce").to_numpy(dtype=np.float32)
    T_zidx = pd.to_numeric(T_locs["zidx"], errors="coerce").to_numpy(dtype=np.float32)

    R_zprec = pd.to_numeric(R_locs["locprecznm"], errors="coerce").to_numpy(dtype=np.float32)
    T_zprec = pd.to_numeric(T_locs["locprecznm"], errors="coerce").to_numpy(dtype=np.float32)

    wR_z = 1.0 / np.maximum(R_zprec[idxR], 1e-6) ** 2
    wT_z = 1.0 / np.maximum(T_zprec[idxT], 1e-6) ** 2
    wsum_z = wR_z + wT_z
    wsum_z[wsum_z == 0.0] = 1.0

    zseed_abs = ((wR_z * R_zidx[idxR]) + (wT_z * T_zidx[idxT])) / wsum_z
    zseed_abs = zseed_abs.astype(np.float32)

    # Reflected-channel ROI centers and residuals in R space.
    R_centers, R_resid = centers_and_residuals(merged_R)

    # Map merged R-space coordinates back into T image space so the T ROI is cut
    # around the corresponding transmitted-channel center.
    Hinv = np.linalg.inv(np.asarray(H_T_to_R, dtype=np.float64))
    merged_T = apply_homography(merged_R.astype(np.float32), Hinv.astype(np.float32))
    T_centers, T_resid = centers_and_residuals(merged_T)

    # Keep only pairs whose R and T ROIs are both fully in bounds.
    _, validR = cut_rois_at_centers(R_phot, R_centers, roi_size)
    _, validT = cut_rois_at_centers(T_phot, T_centers, roi_size)

    valid = validR & validT
    if not np.any(valid):
        return dict(
            frame=frame_idx,
            n_pairs=0,
            rois_R=np.empty((0, roi_size, roi_size), np.float32),
            rois_T=np.empty((0, roi_size, roi_size), np.float32),
            zseed_abs=np.empty((0,), np.float32),
        )

    keep = np.where(valid)[0]

    R_centers_keep = R_centers[keep]
    T_centers_keep = T_centers[keep]

    rois_R, _ = cut_rois_at_centers(R_phot, R_centers_keep, roi_size)
    rois_T, _ = cut_rois_at_centers(T_phot, T_centers_keep, roi_size)

    if rois_R.shape[0] != rois_T.shape[0]:
        raise RuntimeError(
            f"Internal paired ROI mismatch after Stage 1 seeded pairing: "
            f"R={rois_R.shape[0]} T={rois_T.shape[0]}"
        )

    return dict(
        frame=frame_idx,
        n_pairs=int(keep.size),
        rois_R=rois_R,
        rois_T=rois_T,
        R_centers=R_centers_keep,
        T_centers=T_centers_keep,
        R_dxdy=R_resid[keep],
        T_dxdy=T_resid[keep],
        zseed_abs=zseed_abs[keep].astype(np.float32, copy=False),
        pair_idx_R=idxR[keep],
        pair_idx_T=idxT[keep],
    )


def iter_stage3_paired_roi_batches_from_stage1_csvs(
    tif_R_path: Path | str,
    tif_T_path: Path | str,
    R_csv_path: Path | str,
    T_csv_path: Path | str,
    H_T_to_R: np.ndarray,
    *,
    offset: float,
    camera_gain: float,
    QE: float,
    roi_size: int,
    batch_size: int = 4096,
    start_frame: int = 0,
    max_frames: int | None = None,
    pair_max_linf_dist_px: float = 5.0,
    on_frame=None,
):
    """
    Yield GPU-sized batches of paired R/T ROIs for global fitting. (Pipeline stage 3)

    This iterator starts from Stage 1 CSV localizations and existing R/T TIFF
    frames. For each frame it transforms T localizations into R space, pairs R/T
    localizations, merges paired coordinates by inverse-variance weighting,
    preserves per-channel subpixel offsets, cuts paired R/T ROIs, buffers them,
    and yields one GPU-ready paired-ROI batch at a time.

    Yields
    ------
    tuple
        Positional tuple consumed by pipeline_helpers.unpack_stage3_paired_roi_batch:

            rois_R, rois_T
                Image-order ROI cutouts, shape (N, Y, X).
            R_centers, T_centers
                Integer ROI centers, shape (N, 2), stored as [x, y].
            R_dxdy, T_dxdy
                Subpixel residuals, shape (N, 2), stored as [dx, dy].
            zseed_abs
                Absolute spline z-index seeds, shape (N,).
            frames
                Frame index for each pair, shape (N,).
            pair_ids
                Stable pair IDs, shape (N,).
    """
    if roi_size % 2 == 0:
        raise ValueError("roi_size must be odd")

    R_df = _load_stage1_locs_csv(R_csv_path, start_frame=start_frame, max_frames=max_frames)
    T_df = _load_stage1_locs_csv(T_csv_path, start_frame=start_frame, max_frames=max_frames)

    R_by_frame = _group_stage1_locs_by_frame(R_df)
    T_by_frame = _group_stage1_locs_by_frame(T_df)

    pair_id = 0
    buf_R = []; buf_T = []
    buf_Rc = []; buf_Tc = []
    buf_Rd = []; buf_Td = []
    buf_z = []
    buf_f = []; buf_id = []

    H = np.asarray(H_T_to_R, dtype=np.float64)

    _reported_R, nR = count_readable_tiff_prefix(tif_R_path)
    _reported_T, nT = count_readable_tiff_prefix(tif_T_path)
    n_min = min(nR, nT)

    with tifffile.TiffFile(str(tif_R_path)) as tifR, tifffile.TiffFile(str(tif_T_path)) as tifT:
        f0 = int(max(0, start_frame))
        fend = n_min if max_frames is None else min(n_min, f0 + int(max_frames))

        for f in range(f0, fend):
            if on_frame is not None:
                try:
                    on_frame(f)
                except Exception:
                    pass

            R_locs = R_by_frame.get(f)
            T_locs = T_by_frame.get(f)
            if R_locs is None or T_locs is None or R_locs.empty or T_locs.empty:
                continue

            Rraw = safe_read_tiff_frame(tifR, f, tif_path=tif_R_path)
            Traw = safe_read_tiff_frame(tifT, f, tif_path=tif_T_path)

            if Rraw is None or Traw is None:
                # Paired global fitting requires both channels for this frame.
                continue

            R = adu_to_photons(
                Rraw,
                offset=offset,
                camera_gain=camera_gain,
                QE=QE,
            )
            T = adu_to_photons(
                Traw,
                offset=offset,
                camera_gain=camera_gain,
                QE=QE,
            )

            out = combine_and_cut_rois_stage3_from_stage1_one_frame(
                f,
                R,
                T,
                R_locs,
                T_locs,
                H,
                roi_size=roi_size,
                pair_max_linf_dist_px=float(pair_max_linf_dist_px),
            )

            if out["n_pairs"] == 0:
                continue

            buf_R.append(out["rois_R"])
            buf_T.append(out["rois_T"])
            buf_Rc.append(out["R_centers"].astype(np.int32, copy=False))
            buf_Tc.append(out["T_centers"].astype(np.int32, copy=False))
            buf_Rd.append(out["R_dxdy"].astype(np.float32, copy=False))
            buf_Td.append(out["T_dxdy"].astype(np.float32, copy=False))
            buf_z.append(out["zseed_abs"].astype(np.float32, copy=False))

            Np = int(out["n_pairs"])
            buf_f.append(np.full((Np,), f, dtype=np.int32))
            buf_id.append(np.arange(pair_id, pair_id + Np, dtype=np.int64))
            pair_id += Np

            if sum(x.shape[0] for x in buf_R) >= batch_size:
                yield (
                    np.concatenate(buf_R, axis=0),
                    np.concatenate(buf_T, axis=0),
                    np.concatenate(buf_Rc, axis=0),
                    np.concatenate(buf_Tc, axis=0),
                    np.concatenate(buf_Rd, axis=0),
                    np.concatenate(buf_Td, axis=0),
                    np.concatenate(buf_z, axis=0),
                    np.concatenate(buf_f, axis=0),
                    np.concatenate(buf_id, axis=0),
                )
                buf_R.clear(); buf_T.clear()
                buf_Rc.clear(); buf_Tc.clear()
                buf_Rd.clear(); buf_Td.clear()
                buf_z.clear()
                buf_f.clear(); buf_id.clear()

    if buf_R:
        yield (
            np.concatenate(buf_R, axis=0),
            np.concatenate(buf_T, axis=0),
            np.concatenate(buf_Rc, axis=0),
            np.concatenate(buf_Tc, axis=0),
            np.concatenate(buf_Rd, axis=0),
            np.concatenate(buf_Td, axis=0),
            np.concatenate(buf_z, axis=0),
            np.concatenate(buf_f, axis=0),
            np.concatenate(buf_id, axis=0),
        )