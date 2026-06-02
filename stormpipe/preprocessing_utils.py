# preprocessing_utils.py

"""
Preprocessing helpers
---------------------
Shared helpers for TIFF I/O, photon correction, PSF model loading, registration
geometry, candidate-pairing, and ROI cutout.

These utilities are used by both Stage 1 preprocessing and Stage 3 paired-ROI
preparation. Higher-level orchestration lives in the pipeline modules.

TIFF robustness policy
----------------------
Some TIFF files may contain a partially written final frame if the recording cut
out mid-acquisition. tifffile may report the page in len(tif.pages), but fail
when reading its pixel data.

The pipeline treats unreadable frames as missing data:
- trailing unreadable pages are excluded from the nominal readable frame count
- any unreadable frame encountered during streaming is skipped
- paired Stage 3 frames are skipped if either channel cannot be read
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re

import cv2
import numpy as np
import pandas as pd
import scipy.io
import tifffile
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("stormpipe.tiff")



# --- File input/outputs ---
@dataclass(frozen=True)
class ChannelFilePair:
    """
    One reflected/transmitted TIFF pair discovered in an experiment folder.

    stem is the shared, filename-safe acquisition identifier used for Stage 1/2
    intermediates, Stage 3 output folders, and Stage 3 output filenames.

    In flat mode (config: input_search_mode), stem comes from the filename prefix:
        a_ROI-R.tif + a_ROI-T.tif -> stem='a'

    In recursive mode, stem comes from the full relative folder path plus any
    optional filename prefix:
        condition/pos/acq/ROI-R.tif + ROI-T.tif
        -> stem='condition__pos__acq'
    """
    stem: str
    tif_R: Path
    tif_T: Path


def _strip_channel_suffix(stem: str, suffix: str) -> str:
    if not stem.endswith(suffix):
        raise ValueError(f"Stem {stem!r} does not end with suffix {suffix!r}")
    return stem[: -len(suffix)]

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem_component(text: str) -> str:
    """
    Convert one folder-name or filename-prefix component into a safe filename part.
    """
    text = str(text).strip()
    text = text.replace("µ", "u")
    text = _SAFE_STEM_RE.sub("_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    return text or "unnamed"


def _relative_folder_parts(folder: Path, root: Path) -> tuple[str, ...]:
    """
    Return safe relative folder path components from root to folder.
    """
    folder_resolved = Path(folder).resolve(strict=False)
    root_resolved = Path(root).resolve(strict=False)

    try:
        rel = folder_resolved.relative_to(root_resolved)
    except ValueError:
        rel = Path(folder).relative_to(root)

    if str(rel) == ".":
        return ()

    return tuple(
        _safe_stem_component(part)
        for part in rel.parts
        if part not in {"", "."}
    )


def _make_pair_stem(
    *,
    root: Path,
    pair_folder: Path,
    stripped_filename_prefix: str,
    recursive: bool,
) -> str:
    """
    Build a deterministic pair stem from the full relative folder path plus an
    optional stripped filename prefix.
    """
    parts: list[str] = []

    if recursive:
        parts.extend(_relative_folder_parts(pair_folder, root))

    prefix = str(stripped_filename_prefix).strip()
    if prefix:
        parts.append(_safe_stem_component(prefix))

    if not parts:
        # Handles rare cases like data_dir/ROI-R.tif + data_dir/ROI-T.tif.
        parts.append(_safe_stem_component(pair_folder.name))

    return "__".join(parts)

def find_channel_file_pairs(
    folder: Path | str,
    *,
    reflected_suffix: str = "_ROI-R",
    transmitted_suffix: str = "_ROI-T",
    tiff_extension: str = ".tif",
    input_search_mode: str = "flat",
    min_readable_frames_per_channel: int = 0,
) -> list[ChannelFilePair]:
    """
    Discover all matched reflected/transmitted TIFF pairs.

    Discovery modes
    ---------------
    flat
        Search only `folder`.

        Example:
            data_dir/a_ROI-R.tif
            data_dir/a_ROI-T.tif

        -> pair.stem = "a"

    recursive
        Search `folder` and all subfolders. The output pair stem includes all
        relative folder names from `folder` to the folder containing the pair.

        Example:
            data_dir/condition_A/pos_001/acq_001/ROI-R.tif
            data_dir/condition_A/pos_001/acq_001/ROI-T.tif

        -> pair.stem = "condition_A__pos_001__acq_001"

    Pairs are matched only when reflected/transmitted files live in the same
    folder and share the same stripped filename prefix.
    """
    folder = Path(folder)

    reflected_suffix = str(reflected_suffix)
    transmitted_suffix = str(transmitted_suffix)
    tiff_extension = str(tiff_extension)
    input_search_mode = str(input_search_mode)

    if input_search_mode not in {"flat", "recursive"}:
        raise ValueError("input_search_mode must be 'flat' or 'recursive'")

    if not tiff_extension.startswith(".") or len(tiff_extension) < 2:
        raise ValueError("tiff_extension must start with '.', e.g. '.tif'")

    min_frames = int(min_readable_frames_per_channel)
    if min_frames < 0:
        raise ValueError("min_readable_frames_per_channel must be >= 0")

    globber = folder.rglob if input_search_mode == "recursive" else folder.glob

    R_files = sorted(globber(f"*{reflected_suffix}{tiff_extension}"))
    T_files = sorted(globber(f"*{transmitted_suffix}{tiff_extension}"))

    root_resolved = folder.resolve(strict=False)

    def rel_dir_key(path: Path) -> tuple[str, ...]:
        if input_search_mode == "flat":
            return ()
        try:
            rel = path.parent.resolve(strict=False).relative_to(root_resolved)
        except ValueError:
            rel = path.parent.relative_to(folder)
        if str(rel) == ".":
            return ()
        return tuple(str(part) for part in rel.parts)

    def pair_key(path: Path, suffix: str) -> tuple[tuple[str, ...], str]:
        stripped = _strip_channel_suffix(path.stem, suffix)
        return rel_dir_key(path), stripped

    R_by_key = {
        pair_key(p, reflected_suffix): p
        for p in R_files
    }
    T_by_key = {
        pair_key(p, transmitted_suffix): p
        for p in T_files
    }

    R_keys = set(R_by_key)
    T_keys = set(T_by_key)

    missing_T = sorted(R_keys - T_keys)
    missing_R = sorted(T_keys - R_keys)

    def key_label(key: tuple[tuple[str, ...], str]) -> str:
        rel_parts, prefix = key
        rel_text = "/".join(rel_parts)
        if rel_text and prefix:
            return f"{rel_text}/{prefix}"
        if rel_text:
            return rel_text
        if prefix:
            return prefix
        return "<data_dir>"

    if missing_T or missing_R:
        parts = []
        if missing_T:
            parts.append(
                "missing transmitted channel for: "
                + ", ".join(key_label(k) for k in missing_T)
            )
        if missing_R:
            parts.append(
                "missing reflected channel for: "
                + ", ".join(key_label(k) for k in missing_R)
            )
        raise FileNotFoundError(
            "Unmatched reflected/transmitted TIFF files in "
            f"{folder}: " + "; ".join(parts)
        )

    keys = sorted(R_keys & T_keys)
    if not keys:
        raise FileNotFoundError(
            "No matched reflected/transmitted TIFF pairs found in "
            f"{folder} using patterns "
            f"*{reflected_suffix}{tiff_extension} and "
            f"*{transmitted_suffix}{tiff_extension} "
            f"with input_search_mode={input_search_mode!r}"
        )

    pairs: list[ChannelFilePair] = []
    seen_stems: dict[str, tuple[Path, Path]] = {}
    skipped_short: list[tuple[str, int, int]] = []

    for key in keys:
        rel_parts, stripped_prefix = key
        tif_R = R_by_key[key]
        tif_T = T_by_key[key]

        stem = _make_pair_stem(
            root=folder,
            pair_folder=tif_R.parent,
            stripped_filename_prefix=stripped_prefix,
            recursive=(input_search_mode == "recursive"),
        )

        if stem in seen_stems:
            other_R, other_T = seen_stems[stem]
            raise ValueError(
                "Duplicate pair stem generated during TIFF discovery: "
                f"{stem!r}\n"
                f"Existing pair: {other_R} / {other_T}\n"
                f"New pair: {tif_R} / {tif_T}\n"
                "Rename folders/files or adjust suffix settings so pair stems "
                "are unique."
            )

        if min_frames > 0:
            _reported_R, nR = count_readable_tiff_prefix(tif_R)
            _reported_T, nT = count_readable_tiff_prefix(tif_T)

            if min(int(nR), int(nT)) < min_frames:
                logger.info(
                    "Skipping TIFF pair below minimum frame threshold | stem=%s "
                    "| R_readable=%d T_readable=%d min_required=%d",
                    stem,
                    int(nR),
                    int(nT),
                    min_frames,
                )
                skipped_short.append((stem, int(nR), int(nT)))
                continue

        seen_stems[stem] = (tif_R, tif_T)
        pairs.append(
            ChannelFilePair(
                stem=stem,
                tif_R=tif_R,
                tif_T=tif_T,
            )
        )

    if not pairs:
        extra = ""
        if skipped_short:
            extra = (
                f" All {len(skipped_short)} matched pair(s) were skipped because "
                f"they had fewer than {min_frames} readable frames per channel."
            )
        raise FileNotFoundError(
            "No processable reflected/transmitted TIFF pairs found in "
            f"{folder}.{extra}"
        )

    return pairs

def adu_to_photons(
    frame: np.ndarray,
    *,
    offset: float,
    camera_gain: float,
    QE: float,
) -> np.ndarray:
    """
    Photon correction convention:
        photons = max((ADU - offset) * camera_gain * QE, 0)

    Negative photon values are not physically meaningful, so the pipeline clips
    them to zero before detection, ROI extraction, and fitting.
    """
    photons = (
        frame.astype(np.float32, copy=False) - np.float32(offset)
    ) * np.float32(camera_gain) * np.float32(QE)

    return np.maximum(photons, np.float32(0.0)).astype(np.float32, copy=False)

def photon_correct_single_frame(
    tif_path: Path | str,
    frame_index: int,
    *,
    offset: float,
    camera_gain: float,
    QE: float,
) -> np.ndarray:
    """Read a single plane from the TIFF and convert to nonnegative photon units."""
    with tifffile.TiffFile(str(tif_path)) as tif:
        n_pages = len(tif.pages)
        if n_pages == 0:
            raise RuntimeError(f"No frames in {tif_path}")
        f = int(max(0, min(frame_index, n_pages - 1)))
        frame = tif.pages[f].asarray()
    return adu_to_photons(
        frame,
        offset=offset,
        camera_gain=camera_gain,
        QE=QE,
    )

# --- SMAP PSF model loading ---
def load_smap_spline_coeffs(mat_path: str, channel: int = 0, verbose: bool = False) -> dict:
    """
    Load one SMAP spline PSF calibration channel.

    The calibration file may store the three spatial axes in a MATLAB-oriented
    order. This loader identifies the z axis as the longest of the first three
    axes, then returns coefficients in the pipeline's logical calibration layout:

        coeff.shape == (X, Y, Z, B)

    where B is the spline basis dimension, normally 64.

    The returned coefficients are not yet packed for loclib/CUDA. Stage-specific
    GPU helpers perform that boundary conversion later.
    """
    if channel not in (0, 1):
        raise ValueError("Channel must be 0 or 1.")
    mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    try:
        sxy = mat["SXY"]
        cspline = sxy[channel].cspline
    except (KeyError, AttributeError, IndexError) as e:
        raise RuntimeError(f"Could not load SXY[{channel}].cspline from {mat_path}") from e

    smoothed = None
    if channel == 0 and hasattr(cspline, "coeffref"):
        smoothed = cspline.coeffref
    elif channel == 1 and hasattr(cspline, "coeftar"):
        smoothed = cspline.coeftar
    elif hasattr(cspline, "coeff"):
        smoothed = cspline.coeff
    if smoothed is None:
        raw_present = (channel == 0 and hasattr(cspline, "coeffrawref")) or (
            channel == 1 and hasattr(cspline, "coeffrawtar")
        )
        hint = " (file only contains raw coeffs)" if raw_present else ""
        raise RuntimeError(f"No smoothed spline coefficients found in cspline{hint}.")

    coeff = np.asarray(smoothed, dtype=np.float32)
    if coeff.ndim != 4 or coeff.shape[-1] != 64:
        raise ValueError(f"Unexpected coeff shape {coeff.shape}; expected (*,*,*,64).")

    z_axis = int(np.argmax(coeff.shape[:3]))  # longest dimension is usually z
    xy_axes = [a for a in (0, 1, 2) if a != z_axis]
    # Return the logical calibration layout used by the Python pipeline:
    #     coeff(x, y, z, basis64) -> shape (X, Y, Z, B)
    #
    # The xy_axes reversal is part of the SMAP-to-pipeline convention and should
    # remain paired with the downstream loclib coefficient packing helpers.
    perm = (xy_axes[1], xy_axes[0], z_axis, 3)
    coeff = np.ascontiguousarray(coeff.transpose(perm))

    dz = float(getattr(cspline, "dz", 50.0))
    z0 = int(getattr(cspline, "z0", coeff.shape[2] // 2))
    normf = float(getattr(cspline, "normf", 1.0))
    mirror = bool(getattr(cspline, "mirror", False))
    if verbose:
        print(f"Loaded PSF spline from {mat_path}:")
        print(f" - coeff shape: {coeff.shape}")
        print(f" - dz: {dz} nm | z0: {z0} | mirror: {mirror} | normf: {normf:.3f}")

    return {"coeff": coeff, "dz": dz, "z0": z0, "normf": normf, "mirror": mirror}

def load_channel_psf_models(
    calib_source: Path | str,
    channel_map: dict[str, int],
    verbose: bool = True,
) -> dict[str, dict]:
    src = Path(calib_source)
    models: dict[str, dict] = {}
    if src.is_file():
        for tag, ch_idx in channel_map.items():
            model = load_smap_spline_coeffs(str(src), channel=ch_idx, verbose=verbose)
            coeff = model["coeff"]
            if coeff.dtype != np.float32 or coeff.ndim != 4 or coeff.shape[-1] != 64:
                raise ValueError(f"[{tag}] Invalid coeff array from {src}: dtype={coeff.dtype}, shape={coeff.shape}")
            models[tag] = model
        return models

    if src.is_dir():
        file_R = src / "Axcal_inputZStack_cam_R_3dcal.mat"
        file_T = src / "Axcal_inputZStack_cam_T_3dcal.mat"
        for tag, ch_idx in channel_map.items():
            preferred = src / f"Axcal_inputZStack_cam_{tag}_3dcal.mat"
            candidates = []
            if preferred.exists(): candidates.append(preferred)
            other = file_T if tag == "R" else file_R
            if other.exists() and other not in candidates: candidates.append(other)
            for cand in sorted(src.glob("*3dcal.mat")):
                if cand not in candidates: candidates.append(cand)

            last_exc = None
            for cand in candidates:
                try:
                    model = load_smap_spline_coeffs(str(cand), channel=ch_idx, verbose=verbose)
                    coeff = model["coeff"]
                    if coeff.dtype != np.float32 or coeff.ndim != 4 or coeff.shape[-1] != 64:
                        raise ValueError(f"[{tag}] Invalid coeff array: dtype={coeff.dtype}, shape={coeff.shape}")
                    models[tag] = model
                    break
                except Exception as e:
                    last_exc = e
                    continue
            if tag not in models:
                raise RuntimeError(
                    f"Could not load PSF spline for tag '{tag}' (channel {ch_idx}) from {src}. "
                    f"Last error: {last_exc}"
                )
        return models

    raise FileNotFoundError(f"{src} not found")

def safe_read_tiff_frame(
    tif: tifffile.TiffFile,
    frame_index: int,
    *,
    tif_path: Path | str = "",
) -> np.ndarray | None:
    """
    Safely read one TIFF frame as float32.

    Returns None if the page exists in the TIFF directory but cannot be read,
    for example because the acquisition cut out during a partially written final
    frame.
    """
    try:
        return tif.pages[int(frame_index)].asarray().astype(np.float32, copy=False)
    except Exception as e:
        where = f" from {tif_path}" if tif_path else ""
        logger.warning(
            "Skipping unreadable TIFF frame%s | frame=%d | %s",
            where,
            int(frame_index),
            e,
        )
        return None


def count_readable_tiff_prefix(
    tif_path: Path | str,
    *,
    tail_probe_frames: int = 32,
) -> tuple[int, int]:
    """
    Return (reported_pages, readable_prefix_pages).

    This is designed for acquisitions where the final TIFF page may be listed in
    the TIFF directory but contain incomplete image data. The function probes
    backward from the reported end and treats unreadable trailing pages as absent.

    If the tail probe is inconclusive, it falls back to a forward scan until the
    first unreadable frame.
    """
    tif_path = Path(tif_path)

    with tifffile.TiffFile(str(tif_path)) as tif:
        reported = len(tif.pages)

        if reported == 0:
            return 0, 0

        first_to_probe = max(0, reported - int(tail_probe_frames))

        # Fast path: most corruption cases are one or more trailing bad pages.
        for f in range(reported - 1, first_to_probe - 1, -1):
            try:
                _ = tif.pages[f].asarray()
                readable = f + 1

                if readable < reported:
                    logger.warning(
                        "TIFF has unreadable trailing frame(s): %s | "
                        "reported=%d readable_prefix=%d dropped=%d",
                        tif_path,
                        reported,
                        readable,
                        reported - readable,
                    )

                return reported, readable

            except Exception as e:
                logger.warning(
                    "Unreadable TIFF frame during tail probe: %s | frame=%d/%d | %s",
                    tif_path,
                    f,
                    reported - 1,
                    e,
                )

        # Slow fallback: if the probed tail all failed, find the first readable prefix.
        readable = 0
        for f in range(reported):
            try:
                _ = tif.pages[f].asarray()
                readable = f + 1
            except Exception as e:
                logger.warning(
                    "Stopping readable-frame scan at first unreadable TIFF frame: "
                    "%s | frame=%d/%d | %s",
                    tif_path,
                    f,
                    reported - 1,
                    e,
                )
                break

    if readable < reported:
        logger.warning(
            "Using readable TIFF prefix only: %s | reported=%d readable_prefix=%d dropped=%d",
            tif_path,
            reported,
            readable,
            reported - readable,
        )

    return reported, readable

# --- Registration helpers (optional) ---
def apply_homography(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 projective transform to 2D pixel coordinates.

    Parameters
    ----------
    points_xy
        Array of coordinates with shape (N, 2), stored as [x, y].
    H
        Homography matrix. In Stage 3 this is H_T_to_R, mapping transmitted
        coordinates into reflected-channel coordinate space.

    Returns
    -------
    np.ndarray
        Transformed coordinates with shape (N, 2), stored as [x, y].
    """
    points_xy = np.asarray(points_xy)
    pts_homog = np.hstack(
        [points_xy, np.ones((points_xy.shape[0], 1), dtype=points_xy.dtype)]
    )
    pts_transformed = (H @ pts_homog.T).T
    pts_transformed /= pts_transformed[:, 2:3]
    return pts_transformed[:, :2]

def estimate_projective_transform(src_pts: np.ndarray, dst_pts: np.ndarray):
    H, mask = cv2.findHomography(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H, mask

def match_nearest_neighbors(df_ref: pd.DataFrame, df_target: pd.DataFrame, max_distance: float):
    matched_ref = []
    matched_target = []
    frames = np.intersect1d(df_ref["frame"].unique(), df_target["frame"].unique())
    for frame in frames:
        ref_pts = df_ref[df_ref["frame"] == frame][["x_pix", "y_pix"]].to_numpy(dtype=np.float32)
        tgt_pts = df_target[df_target["frame"] == frame][["x_pix", "y_pix"]].to_numpy(dtype=np.float32)
        if len(ref_pts) == 0 or len(tgt_pts) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=1).fit(ref_pts)
        distances, indices = nn.kneighbors(tgt_pts)
        for i, d in enumerate(distances[:, 0]):
            if d < max_distance:
                matched_ref.append(ref_pts[indices[i, 0]])
                matched_target.append(tgt_pts[i])
    return np.vstack(matched_ref), np.vstack(matched_target)

# --- Misc ---
def get_dynamic_cutoff(DoG: np.ndarray, threshold_factor: float = 1.0) -> np.float32:
    intensities = DoG[DoG > 0]
    if intensities.size < 10:
        return np.float32(np.mean(intensities) * threshold_factor if intensities.size > 0 else 0)
    p20, p50, p80 = np.percentile(intensities, [20, 50, 80]).astype(np.float32)
    slope = (p80 - p20) / 0.6
    return np.float32(p50 + slope * threshold_factor)

def pair_mutual_linf_one_to_one(
    R_pts: np.ndarray,      # (NR,2) float32/64  in R space
    Tpts_in_R: np.ndarray,  # (NT,2) float32/64  T seeds already mapped into R space
    max_linf_dist: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Conservatively pair same-frame R and T localizations in reflected-channel space.

    Pairing rule
    ------------
    1. Use L∞ distance, so a candidate must satisfy both |dx| <= r and |dy| <= r.
    2. Keep only mutual nearest-neighbor matches.
    3. Reject clustered neighborhoods: if either member of a proposed pair has any
    other candidate from the opposite channel inside the gate, drop the pair.

    Parameters
    ----------
    R_pts
        Reflected-channel coordinates in R space, shape (NR, 2), stored as [x, y].
    Tpts_in_R
        Transmitted-channel coordinates already transformed into R space,
        shape (NT, 2), stored as [x, y].
    max_linf_dist
        Maximum allowed L∞ distance in pixels.

    Returns
    -------
    idxR, idxT
        Matched indices into R_pts and Tpts_in_R.
    """
    if R_pts.size == 0 or Tpts_in_R.size == 0:
        return np.empty((0,), np.int32), np.empty((0,), np.int32)

    R = np.asarray(R_pts, dtype=np.float32)
    T = np.asarray(Tpts_in_R, dtype=np.float32)

    # L∞ distances (vectorized in chunks to keep memory tame)
    def knn_Linf(A, B):
        idx = np.empty(A.shape[0], dtype=np.int32)
        dist = np.empty(A.shape[0], dtype=np.float32)
        CH = max(1, 8192 // max(1, B.shape[0]))  # simple chunking
        for i0 in range(0, A.shape[0], CH):
            i1 = min(A.shape[0], i0 + CH)
            a = A[i0:i1, None, :]      # (m,1,2)
            b = B[None, :, :]          # (1,n,2)
            d = np.max(np.abs(a - b), axis=2)  # (m,n) L∞
            j = np.argmin(d, axis=1)
            idx[i0:i1] = j
            dist[i0:i1] = d[np.arange(j.size), j]
        return idx, dist

    t2r_idx, t2r_d = knn_Linf(T, R)
    r2t_idx, r2t_d = knn_Linf(R, T)

    # Mutual matches within L∞ gate
    pairs = [(iR, iT)
             for iR, iT in enumerate(r2t_idx)
             if (r2t_d[iR] <= max_linf_dist) and (t2r_idx[iT] == iR) and (t2r_d[iT] <= max_linf_dist)]

    if not pairs:
        return np.empty((0,), np.int32), np.empty((0,), np.int32)

    # Cluster rejection: discard neighborhoods with >2 points (more-than-pairs)
    # Rule: for a proposed pair (iR,iT), if R[iR] has any *other* T within gate OR
    # T[iT] has any *other* R within gate → mark as clustered → drop.
    Rxy = R
    Txy = T
    keep = []
    r_gate = float(max_linf_dist)
    for (iR, iT) in pairs:
        # count neighbors of R[iR] among T within gate (excluding matched iT)
        dT = np.max(np.abs(Txy - Rxy[iR]), axis=1)
        neighT = np.where((dT <= r_gate))[0]
        neighT = neighT[neighT != iT]
        # count neighbors of T[iT] among R within gate (excluding matched iR)
        dR = np.max(np.abs(Rxy - Txy[iT]), axis=1)
        neighR = np.where((dR <= r_gate))[0]
        neighR = neighR[neighR != iR]
        clustered = (neighT.size > 0) or (neighR.size > 0)
        if not clustered:
            keep.append((iR, iT))

    if not keep:
        return np.empty((0,), np.int32), np.empty((0,), np.int32)

    idxR = np.asarray([i for (i, _) in keep], dtype=np.int32)
    idxT = np.asarray([j for (_, j) in keep], dtype=np.int32)
    return idxR, idxT


def centers_and_residuals(coords_float: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Round continuous [x, y] pixel coordinates to integer ROI centers.

    Parameters
    ----------
    coords_float
        Continuous full-frame coordinates with shape (N, 2), stored as [x, y].

    Returns
    -------
    centers_int
        Rounded integer ROI centers with shape (N, 2), stored as [x, y].
    residuals
        Subpixel residuals with shape (N, 2), stored as [dx, dy]:

            residual = coords_float - centers_int
    """
    c = np.asarray(coords_float, dtype=np.float32)
    centers = np.rint(c).astype(np.int32)
    residuals = c - centers.astype(np.float32)
    return centers, residuals


def cut_rois_at_centers(image: np.ndarray, centers_int: np.ndarray, roi_size: int):
    """
    Cut image-order ROIs around integer [x, y] centers.

    Parameters
    ----------
    image
        Single-channel photon-corrected image with shape (Y, X).
    centers_int
        Full-frame integer ROI centers with shape (N, 2), stored as [x, y].
    roi_size
        Odd ROI side length.

    Returns
    -------
    rois
        Image-order ROI cutouts with shape (M, K, K), equivalent to (M, Y, X).
        These are later converted to loclib layout by GPU helper functions.
    valid
        Boolean mask with shape (N,), True when the ROI was fully in bounds.
    """
    Y, X = image.shape
    K = int(roi_size)
    assert K % 2 == 1, "roi_size must be odd"
    half = (K - 1) // 2

    if centers_int.size == 0:
        return np.empty((0, K, K), np.float32), np.zeros((0,), bool)

    x = centers_int[:, 0].astype(np.int32)
    y = centers_int[:, 1].astype(np.int32)

    valid = (y >= half) & (y < Y - half) & (x >= half) & (x < X - half)
    if not np.any(valid):
        return np.empty((0, K, K), np.float32), valid

    xv = x[valid]; yv = y[valid]
    win = sliding_window_view(image, (K, K))              # (H-K+1, W-K+1, K, K)
    rois_view = win[yv - half, xv - half]                 # (M, K, K)
    rois = np.ascontiguousarray(rois_view, dtype=np.float32)
    return rois, valid


