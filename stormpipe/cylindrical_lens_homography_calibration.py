# cylindrical_lens_homography_calibration.py

"""
Cylindrical-lens field-distortion homography calibration.

This standalone calibration workflow estimates a homography:

    H_cyl_to_nocyl

mapping bead coordinates recorded with the cylindrical lens in place into the
reference coordinate system recorded without the cylindrical lens.

Workflow
--------
1. Discover paired calibration TIFF movies in a folder:
       <stem>_cylindrical_lens.tif
       <stem>_no_cylindrical_lens.tif

2. For each pair:
       cylindrical-lens movie:
           spline-smoothed astigmatic PSF fit using existing Stage 1 fitter

       no-cylindrical-lens movie:
           simple 2D Gaussian MLE fit using loclib single-channel Gaussian mode

       both localization tables:
           spatially cluster repeated bead localizations into bead centroids

       centroid tables:
           mutual nearest-neighbor pair cylindrical/no-cylindrical bead centroids

3. Pool bead correspondences across all calibration pairs.

4. Estimate one global RANSAC homography:
       cylindrical-lens coordinates -> no-cylindrical-lens coordinates

5. Save homography, centroid tables, bead-pair tables, pooled correspondences,
   and summary diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re

import cv2
import numpy as np
import pandas as pd

from .loclib_ctypes import localizationlib

from .pipeline_config import Settings, psf_model_format_normalized
from .mle_fitting_drivers import (
    ChannelModel,
    fit_stage1_single_channel,
)
from .mle_fitting_helpers import (
    build_single_channel_splinesize_for_dll,
    coeff_xyzb_matlab_to_single_channel_loclib_bzyx,
    rois_nyx_image_to_loclib_nxy,
)
from .roi_preparation import iter_single_channel_roi_batches_from_tiff
from .preprocessing_utils import apply_homography, load_channel_psf_models
from .stage3_pipeline_helpers import CsvBatchWriter
from .uipsf_model_loader import load_uipsf_coeff_tensor


logger = logging.getLogger("stormpipe.cyl_lens_calibration")


_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class CylindricalLensCalibrationSettings:
    # -------------------------------------------------------------------------
    # Required paths
    # -------------------------------------------------------------------------
    calibration_dir: Path
    output_dir: Path
    psf_model: Path

    # -------------------------------------------------------------------------
    # Pair discovery
    # -------------------------------------------------------------------------
    cylindrical_suffix: str = "_cylindrical_lens"
    no_cylindrical_suffix: str = "_no_cylindrical_lens"
    tiff_extension: str = ".tif"
    input_search_mode: str = "flat"  # "flat" or "recursive"

    # -------------------------------------------------------------------------
    # Channel / model choice
    # -------------------------------------------------------------------------
    channel_tag: str = "R"  # "R" or "T"
    channel_map: dict[str, int] | None = None

    # PSF model file format:
    #   "SMAP"  -> MATLAB/SMAP .mat model
    #   "uiPSF" -> Python uiPSF .h5 model
    psf_model_format: str = "SMAP"

    # uiPSF-only options. Ignored when psf_model_format="SMAP".
    uipsf_coeff_key: str = "coeff"
    uipsf_z0_index: int | None = None
    uipsf_swap_xy_axes: bool = False

    # -------------------------------------------------------------------------
    # Camera correction
    # -------------------------------------------------------------------------
    pixelsize_nm: float = 97.0
    offset: float = 100.0
    camera_gain: float = 0.23
    qe: float = 1.0

    # -------------------------------------------------------------------------
    # Detection / ROI extraction
    # -------------------------------------------------------------------------
    roi_size: int = 13
    thr_factor_cylindrical: float = 2.0
    thr_factor_no_cylindrical: float = 2.0
    sigma1: float = 1.2
    sigma2: float = 3.2
    min_distance: int = 5
    keep_peak_winner: bool = True

    # Optional cap for quick tests. None = use all readable frames.
    max_frames_per_movie: int | None = None

    # -------------------------------------------------------------------------
    # GPU fitting
    # -------------------------------------------------------------------------
    usecuda: int = 1
    gpu_iterations: int = 150
    gpu_batch_size: int = 9000
    em_excess_noise: int = 1
    ri_mismatch: float = 1.0

    # Gaussian fit settings for no-cylindrical-lens movies.
    gaussian_free_sigma: bool = True
    gaussian_fixed_sigma_px: float = 1.5
    gaussian_iterations: int = 100

    # -------------------------------------------------------------------------
    # Bead centroiding
    # -------------------------------------------------------------------------
    cluster_eps_px: float = 1.5
    min_detections_per_bead: int = 20
    centroid_method: str = "median"  # "median" or "weighted_mean"

    # Optional localization filters before bead clustering.
    centroid_max_locprec_nm: float | None = None
    centroid_min_photons: float | None = None
    require_converged_for_centroids: bool = True

    # -------------------------------------------------------------------------
    # Bead pairing / homography
    # -------------------------------------------------------------------------
    max_centroid_pair_dist_px: float = 20.0
    ransac_reproj_thresh_px: float = 2.0
    min_pairs_for_homography: int = 4

    # -------------------------------------------------------------------------
    # Output organization
    # -------------------------------------------------------------------------
    # The final text homography is written directly in output_dir.
    # All CSV diagnostics, pair-level outputs, and the NPY copy are written under
    # output_dir / diagnostics_dir_name.
    homography_txt_name: str = "cyl_lens_correction_projective_homography.txt"
    homography_npy_name: str = "cyl_lens_correction_projective_homography.npy"
    diagnostics_dir_name: str = "calibration_diagnostics"

    # -------------------------------------------------------------------------
    # Run behavior
    # -------------------------------------------------------------------------
    overwrite_existing_locs: bool = True
    skip_failed_pairs: bool = True

    def pipeline_settings_for_spline_fit(self) -> Settings:
        """
        Build a normal pipeline Settings object so the calibration script can
        reuse the existing Stage 1 spline fitter for cylindrical-lens movies.
        """
        channel_map = self.channel_map
        if channel_map is None:
            channel_map = {"R": 0, "T": 1}

        return Settings(
            data_dir=Path(self.calibration_dir),
            psf_model=Path(self.psf_model),
            psf_model_format=str(self.psf_model_format),

            uipsf_coeff_key=str(self.uipsf_coeff_key),
            uipsf_z0_index=self.uipsf_z0_index,
            uipsf_swap_xy_axes=bool(self.uipsf_swap_xy_axes),

            pipeline_mode="full",
            channel_map=channel_map,
            input_search_mode=self.input_search_mode,
            min_readable_frames_per_channel=0,
            pixelsize_nm=float(self.pixelsize_nm),
            offset=float(self.offset),
            camera_gain=float(self.camera_gain),
            qe=float(self.qe),
            roi_size=int(self.roi_size),
            thr_factor_reflected=float(self.thr_factor_cylindrical),
            thr_factor_transmitted=float(self.thr_factor_cylindrical),
            sigma1=float(self.sigma1),
            sigma2=float(self.sigma2),
            min_distance=int(self.min_distance),
            keep_peak_winner=bool(self.keep_peak_winner),
            gpu_iterations=int(self.gpu_iterations),
            gpu_batch_size=int(self.gpu_batch_size),
            em_excess_noise=int(self.em_excess_noise),
            ri_mismatch=float(self.ri_mismatch),
        )


@dataclass(frozen=True)
class CylindricalLensMoviePair:
    stem: str
    cylindrical_tif: Path
    no_cylindrical_tif: Path

def _normalized_channel_tag(settings: CylindricalLensCalibrationSettings) -> str:
    tag = str(settings.channel_tag).strip().upper()
    if tag not in {"R", "T"}:
        raise ValueError(f"channel_tag must be 'R' or 'T', got {settings.channel_tag!r}")
    return tag


def _channel_map_for_settings(settings: CylindricalLensCalibrationSettings) -> dict[str, int]:
    if settings.channel_map is None:
        return {"R": 0, "T": 1}
    return dict(settings.channel_map)


def prepare_calibration_spline_channel_model(
    settings: CylindricalLensCalibrationSettings,
    *,
    pipeline_settings: Settings | None = None,
) -> ChannelModel:
    """
    Prepare only the single spline PSF model needed for the cylindrical-lens
    calibration movie.

    This avoids requiring both R and T models when the calibration only fits one
    channel.
    """
    tag = _normalized_channel_tag(settings)

    channel_map = _channel_map_for_settings(settings)
    if tag not in channel_map:
        raise ValueError(
            f"channel_map must contain an entry for channel_tag={tag!r}; "
            f"got keys {sorted(channel_map)}"
        )

    ch_idx = int(channel_map[tag])

    if pipeline_settings is None:
        pipeline_settings = settings.pipeline_settings_for_spline_fit()

    fmt = psf_model_format_normalized(pipeline_settings)

    if fmt == "SMAP":
        psf = load_channel_psf_models(
            pipeline_settings.psf_model,
            {tag: ch_idx},
            verbose=False,
        )

        psf[tag]["mirror"] = False

        coeff_loclib = coeff_xyzb_matlab_to_single_channel_loclib_bzyx(
            psf[tag]["coeff"]
        )
        splinesize = build_single_channel_splinesize_for_dll(coeff_loclib)

        return ChannelModel(
            coeff=np.ascontiguousarray(coeff_loclib, dtype=np.float32),
            splinesize=np.asarray(splinesize, dtype=np.int32),
            zseed=np.float32(psf[tag]["z0"] + 1e-6),
            dz=float(psf[tag]["dz"]),
            z0=int(psf[tag]["z0"]),
            mirror=bool(psf[tag]["mirror"]),
            normf=float(psf[tag]["normf"]),
        )

    if fmt == "uiPSF":
        data = load_uipsf_coeff_tensor(
            pipeline_settings.psf_model,
            coeff_key=pipeline_settings.uipsf_coeff_key,
            z0_index=pipeline_settings.uipsf_z0_index,
            swap_xy_axes=pipeline_settings.uipsf_swap_xy_axes,
            verbose=True,
        )

        if ch_idx not in (0, 1):
            raise ValueError(
                f"uiPSF calibration supports channel indices 0 and 1 only; "
                f"channel_map[{tag!r}]={ch_idx}"
            )

        coeff_loclib = np.ascontiguousarray(data.coeff[ch_idx], dtype=np.float32)
        splinesize = build_single_channel_splinesize_for_dll(coeff_loclib)

        return ChannelModel(
            coeff=coeff_loclib,
            splinesize=np.asarray(splinesize, dtype=np.int32),
            zseed=np.float32(data.zseed),
            dz=float(data.dz),
            z0=int(data.z0),
            mirror=bool(data.mirror),
            normf=float(data.normf[ch_idx]),
        )

    raise ValueError(f"Unsupported psf_model_format={fmt!r}")

def _safe_stem_component(text: str) -> str:
    text = str(text).strip().replace("µ", "u")
    text = _SAFE_STEM_RE.sub("_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    return text or "unnamed"


def _relative_folder_parts(folder: Path, root: Path) -> tuple[str, ...]:
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
    parts: list[str] = []

    if recursive:
        parts.extend(_relative_folder_parts(pair_folder, root))

    prefix = str(stripped_filename_prefix).strip()
    if prefix:
        parts.append(_safe_stem_component(prefix))

    if not parts:
        parts.append(_safe_stem_component(pair_folder.name))

    return "__".join(parts)


def find_cylindrical_lens_movie_pairs(
    folder: Path | str,
    *,
    cylindrical_suffix: str,
    no_cylindrical_suffix: str,
    tiff_extension: str = ".tif",
    input_search_mode: str = "flat",
) -> list[CylindricalLensMoviePair]:
    """
    Discover paired cylindrical/no-cylindrical calibration TIFF movies.

    Important suffix note
    ---------------------
    The default no-cylindrical suffix, "_no_cylindrical_lens", itself ends with
    "_cylindrical_lens". Therefore discovery classifies no-cylindrical files first.
    """
    folder = Path(folder)

    if input_search_mode not in {"flat", "recursive"}:
        raise ValueError("input_search_mode must be 'flat' or 'recursive'")
    if not tiff_extension.startswith("."):
        raise ValueError("tiff_extension must start with '.', e.g. '.tif'")
    if cylindrical_suffix == no_cylindrical_suffix:
        raise ValueError("cylindrical_suffix and no_cylindrical_suffix must differ")

    globber = folder.rglob if input_search_mode == "recursive" else folder.glob
    files = sorted(globber(f"*{tiff_extension}"))

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

    cyl_by_key: dict[tuple[tuple[str, ...], str], Path] = {}
    nocyl_by_key: dict[tuple[tuple[str, ...], str], Path] = {}

    for p in files:
        stem = p.stem

        if stem.endswith(no_cylindrical_suffix):
            prefix = stem[: -len(no_cylindrical_suffix)]
            nocyl_by_key[(rel_dir_key(p), prefix)] = p
        elif stem.endswith(cylindrical_suffix):
            prefix = stem[: -len(cylindrical_suffix)]
            cyl_by_key[(rel_dir_key(p), prefix)] = p

    cyl_keys = set(cyl_by_key)
    nocyl_keys = set(nocyl_by_key)

    missing_nocyl = sorted(cyl_keys - nocyl_keys)
    missing_cyl = sorted(nocyl_keys - cyl_keys)

    def key_label(key: tuple[tuple[str, ...], str]) -> str:
        rel_parts, prefix = key
        rel_text = "/".join(rel_parts)
        if rel_text and prefix:
            return f"{rel_text}/{prefix}"
        return rel_text or prefix or "<calibration_dir>"

    if missing_nocyl or missing_cyl:
        parts = []
        if missing_nocyl:
            parts.append(
                "missing no-cylindrical-lens movie for: "
                + ", ".join(key_label(k) for k in missing_nocyl)
            )
        if missing_cyl:
            parts.append(
                "missing cylindrical-lens movie for: "
                + ", ".join(key_label(k) for k in missing_cyl)
            )
        raise FileNotFoundError(
            f"Unmatched cylindrical/no-cylindrical TIFF files in {folder}: "
            + "; ".join(parts)
        )

    keys = sorted(cyl_keys & nocyl_keys)
    if not keys:
        raise FileNotFoundError(
            f"No paired calibration TIFFs found in {folder} using suffixes "
            f"{cylindrical_suffix!r} and {no_cylindrical_suffix!r}"
        )

    pairs: list[CylindricalLensMoviePair] = []
    seen_stems: dict[str, tuple[Path, Path]] = {}

    for key in keys:
        rel_parts, prefix = key
        cyl = cyl_by_key[key]
        nocyl = nocyl_by_key[key]

        pair_stem = _make_pair_stem(
            root=folder,
            pair_folder=cyl.parent,
            stripped_filename_prefix=prefix,
            recursive=(input_search_mode == "recursive"),
        )

        if pair_stem in seen_stems:
            old_cyl, old_nocyl = seen_stems[pair_stem]
            raise ValueError(
                f"Duplicate calibration pair stem generated: {pair_stem!r}\n"
                f"Existing: {old_cyl} / {old_nocyl}\n"
                f"New: {cyl} / {nocyl}"
            )

        seen_stems[pair_stem] = (cyl, nocyl)
        pairs.append(
            CylindricalLensMoviePair(
                stem=pair_stem,
                cylindrical_tif=cyl,
                no_cylindrical_tif=nocyl,
            )
        )

    return pairs


def mle_gaussian_single_channel(
    dll,
    rois_nyx: np.ndarray,
    *,
    free_sigma: bool = True,
    fixed_sigma_px: float = 1.5,
    iterations: int = 100,
    em_excess_noise: int = 1,
):
    """
    Run loclib single-channel Gaussian MLE on image-order ROIs.

    Input ROIs follow the pipeline convention:
        rois_nyx.shape == (N, Y, X)

    The wrapper converts them to the same loclib memory-layout boundary used by
    the existing single-channel spline wrapper.
    """
    rois_nyx = np.asarray(rois_nyx, dtype=np.float32)

    if rois_nyx.ndim != 3:
        raise ValueError(f"rois_nyx must have shape (N,Y,X), got {rois_nyx.shape}")

    N, Ysz, Xsz = map(int, rois_nyx.shape)

    if N == 0:
        return (
            np.zeros((6, 0), dtype=np.float32),
            np.zeros((5, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            0,
        )

    rois_loclib_nxy = rois_nyx_image_to_loclib_nxy(rois_nyx)

    em = np.float32(em_excess_noise if em_excess_noise else 1.0)
    data = np.ascontiguousarray(rois_loclib_nxy / em, dtype=np.float32)

    fittype = np.int32(2 if free_sigma else 1)
    coeff = np.array([float(fixed_sigma_px)], dtype=np.float32)
    varim = np.array(0, dtype=np.float32)
    init_z = np.float32(0.0)

    # Square ROIs mean X/Y values are usually identical, but keep the same
    # convention as the existing single-channel spline wrapper.
    datasize = np.array([Ysz, Xsz, N], dtype=np.int32)
    splinesize = np.array([0], dtype=np.int32)

    P = np.zeros((6, N), dtype=np.float32)
    CRLB = np.zeros((5, N), dtype=np.float32)
    LL = np.zeros((N,), dtype=np.float32)

    dll._mleFit(
        data,
        fittype,
        np.int32(int(iterations)),
        coeff,
        varim,
        init_z,
        datasize,
        splinesize,
        P,
        CRLB,
        LL,
    )

    return P, CRLB, LL, N


def format_gaussian_fit_results(
    P: np.ndarray,
    CRLB: np.ndarray,
    LL: np.ndarray,
    xpix_seed: np.ndarray,
    ypix_seed: np.ndarray,
    frames: np.ndarray,
    roi_ids: np.ndarray,
    *,
    roi_size: int,
    pixelsize_nm: float,
    em_excess_noise: int = 1,
) -> dict[str, np.ndarray]:
    """
    Convert raw loclib Gaussian MLE outputs into a compact localization table.

    Loclib parameter convention is treated as:
        P[0] = local y
        P[1] = local x
        P[2] = photons
        P[3] = background
        P[4] = sigma / extra Gaussian parameter
        P[5] = iterations
    """
    P = np.asarray(P, dtype=np.float32)
    CRLB = np.asarray(CRLB, dtype=np.float32)
    LL = np.asarray(LL, dtype=np.float32)

    CRLB = np.nan_to_num(CRLB, nan=0.0)
    CRLB = np.maximum(CRLB, 0.0)

    Yc = P[0]
    Xc = P[1]
    Phot = P[2]
    BG = P[3]
    Sigma = P[4]
    Iter = P[5]

    half = np.float32((int(roi_size) - 1) // 2)

    seed_x = xpix_seed.astype(np.float32, copy=False)
    seed_y = ypix_seed.astype(np.float32, copy=False)

    xpix = seed_x + (Xc - half)
    ypix = seed_y + (Yc - half)

    px_nm = np.float32(pixelsize_nm)
    xnm = (xpix + np.float32(1.0)) * px_nm
    ynm = (ypix + np.float32(1.0)) * px_nm

    xerr_pix = np.sqrt(np.maximum(CRLB[1], 0.0)).astype(np.float32)
    yerr_pix = np.sqrt(np.maximum(CRLB[0], 0.0)).astype(np.float32)
    locprecnm = (
        np.sqrt(0.5 * (xerr_pix**2 + yerr_pix**2)) * px_nm
    ).astype(np.float32)

    center = np.float32((int(roi_size) - 1) / 2.0)
    halfwidth = np.float32(int(roi_size) / 4.0)
    lo = center - halfwidth
    hi = center + halfwidth
    eps = np.float32(1e-3)
    edge_clamped = (
        (np.abs(Xc - lo) <= eps)
        | (np.abs(Xc - hi) <= eps)
        | (np.abs(Yc - lo) <= eps)
        | (np.abs(Yc - hi) <= eps)
    )

    em = np.float32(em_excess_noise if em_excess_noise else 1.0)

    return {
        "roi_id": roi_ids.astype(np.int64, copy=False),
        "frame": frames.astype(np.int32, copy=False),
        "xpix_seed": xpix_seed.astype(np.int32, copy=False),
        "ypix_seed": ypix_seed.astype(np.int32, copy=False),
        "xpix": xpix.astype(np.float32),
        "ypix": ypix.astype(np.float32),
        "xnm": xnm.astype(np.float32),
        "ynm": ynm.astype(np.float32),
        "photons": (Phot * em).astype(np.float32),
        "bg": (BG * em).astype(np.float32),
        "sigma": Sigma.astype(np.float32),
        "xerr": xerr_pix,
        "yerr": yerr_pix,
        "locprecnm": locprecnm,
        "logL": LL.astype(np.float32),
        "iterations": Iter.astype(np.float32),
        "edge_clamped": edge_clamped.astype(bool),
    }


def fit_no_cylindrical_movie_gaussian(
    tif_path: Path,
    out_csv: Path,
    settings: CylindricalLensCalibrationSettings,
    dll,
) -> Path:
    """
    Fit the no-cylindrical-lens bead movie using loclib 2D Gaussian MLE.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    writer = CsvBatchWriter(out_csv, flush_every=10)

    total_rois = 0

    for rois, xpix, ypix, frames, roi_ids in iter_single_channel_roi_batches_from_tiff(
        tif_path,
        offset=float(settings.offset),
        camera_gain=float(settings.camera_gain),
        QE=float(settings.qe),
        roi_size=int(settings.roi_size),
        thr=float(settings.thr_factor_no_cylindrical),
        s1=float(settings.sigma1),
        s2=float(settings.sigma2),
        keep_peak_winner=bool(settings.keep_peak_winner),
        mindist=int(settings.min_distance),
        batch_size=int(settings.gpu_batch_size),
        max_frames=settings.max_frames_per_movie,
        start_frame=0,
        roi_id_start=0,
    ):
        if rois.shape[0] == 0:
            continue

        P, CRLB, LL, _ = mle_gaussian_single_channel(
            dll,
            rois,
            free_sigma=bool(settings.gaussian_free_sigma),
            fixed_sigma_px=float(settings.gaussian_fixed_sigma_px),
            iterations=int(settings.gaussian_iterations),
            em_excess_noise=int(settings.em_excess_noise),
        )

        cols = format_gaussian_fit_results(
            P,
            CRLB,
            LL,
            xpix,
            ypix,
            frames,
            roi_ids,
            roi_size=int(settings.roi_size),
            pixelsize_nm=float(settings.pixelsize_nm),
            em_excess_noise=int(settings.em_excess_noise),
        )

        df = pd.DataFrame(cols)
        writer.write(df)
        total_rois += int(len(df))

    writer.finalize()

    logger.info(
        "No-cylindrical Gaussian fit complete: %s | rows=%d",
        out_csv,
        total_rois,
    )
    return out_csv


def _filter_locs_for_centroids(
    locs: pd.DataFrame,
    settings: CylindricalLensCalibrationSettings,
    *,
    iteration_cap: int | None = None,
) -> pd.DataFrame:
    if locs is None or locs.empty:
        return pd.DataFrame(columns=list(locs.columns) if locs is not None else [])

    df = locs.copy()

    required = {"xpix", "ypix"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Localization table missing required centroid columns: {sorted(missing)}")

    df["xpix"] = pd.to_numeric(df["xpix"], errors="coerce")
    df["ypix"] = pd.to_numeric(df["ypix"], errors="coerce")
    df = df.dropna(subset=["xpix", "ypix"]).copy()

    if "edge_clamped" in df.columns:
        df = df.loc[~df["edge_clamped"].astype(bool)].copy()

    if settings.centroid_max_locprec_nm is not None and "locprecnm" in df.columns:
        df["locprecnm"] = pd.to_numeric(df["locprecnm"], errors="coerce")
        df = df.loc[df["locprecnm"] <= float(settings.centroid_max_locprec_nm)].copy()

    if settings.centroid_min_photons is not None and "photons" in df.columns:
        df["photons"] = pd.to_numeric(df["photons"], errors="coerce")
        df = df.loc[df["photons"] >= float(settings.centroid_min_photons)].copy()

    if (
        settings.require_converged_for_centroids
        and "iterations" in df.columns
    ):
        df["iterations"] = pd.to_numeric(df["iterations"], errors="coerce")

        if iteration_cap is None:
            max_iter = max(int(settings.gpu_iterations), int(settings.gaussian_iterations))
        else:
            max_iter = int(iteration_cap)

        df = df.loc[df["iterations"] < max_iter].copy()

    return df.reset_index(drop=True)


def localizations_to_bead_centroids(
    locs: pd.DataFrame,
    settings: CylindricalLensCalibrationSettings,
    *,
    source_label: str,
) -> pd.DataFrame:
    """
    Cluster repeated localizations into bead centroids.
    """
    from sklearn.cluster import DBSCAN

    if source_label == "cylindrical":
        iteration_cap = int(settings.gpu_iterations)
    elif source_label == "no_cylindrical":
        iteration_cap = int(settings.gaussian_iterations)
    else:
        iteration_cap = None

    df = _filter_locs_for_centroids(
        locs,
        settings,
        iteration_cap=iteration_cap,
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "bead_id",
                "source",
                "xpix",
                "ypix",
                "n_detections",
                "x_std_px",
                "y_std_px",
            ]
        )

    xy = df[["xpix", "ypix"]].to_numpy(dtype=np.float64)

    labels = DBSCAN(
        eps=float(settings.cluster_eps_px),
        min_samples=int(settings.min_detections_per_bead),
    ).fit_predict(xy)

    df = df.copy()
    df["bead_cluster"] = labels

    rows: list[dict] = []
    bead_id = 0

    for label, g in df.loc[df["bead_cluster"] >= 0].groupby("bead_cluster", sort=True):
        if len(g) < int(settings.min_detections_per_bead):
            continue

        x = g["xpix"].to_numpy(dtype=np.float64)
        y = g["ypix"].to_numpy(dtype=np.float64)

        if (
            settings.centroid_method == "weighted_mean"
            and "locprecnm" in g.columns
        ):
            lp = pd.to_numeric(g["locprecnm"], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(lp) & (lp > 0)
            if np.any(valid):
                w = np.zeros(len(g), dtype=np.float64)
                w[valid] = 1.0 / (lp[valid] * lp[valid])
                x_c = float(np.sum(x * w) / np.sum(w))
                y_c = float(np.sum(y * w) / np.sum(w))
            else:
                x_c = float(np.nanmedian(x))
                y_c = float(np.nanmedian(y))
        elif settings.centroid_method == "median":
            x_c = float(np.nanmedian(x))
            y_c = float(np.nanmedian(y))
        else:
            raise ValueError("centroid_method must be 'median' or 'weighted_mean'")

        bead_id += 1
        rows.append(
            {
                "bead_id": bead_id,
                "source": source_label,
                "xpix": x_c,
                "ypix": y_c,
                "xnm": (x_c + 1.0) * float(settings.pixelsize_nm),
                "ynm": (y_c + 1.0) * float(settings.pixelsize_nm),
                "n_detections": int(len(g)),
                "x_std_px": float(np.nanstd(x)),
                "y_std_px": float(np.nanstd(y)),
                "cluster_label": int(label),
            }
        )

    return pd.DataFrame(rows)


def _nearest_neighbor_indices(
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each row in A, find nearest row in B.
    Returns nearest indices and Euclidean distances.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.size == 0 or B.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float64)

    idx = np.empty(A.shape[0], dtype=np.int32)
    dist = np.empty(A.shape[0], dtype=np.float64)

    chunk = max(1, 2000 // max(1, B.shape[0]))

    for i0 in range(0, A.shape[0], chunk):
        i1 = min(A.shape[0], i0 + chunk)
        d2 = np.sum((A[i0:i1, None, :] - B[None, :, :]) ** 2, axis=2)
        j = np.argmin(d2, axis=1)
        idx[i0:i1] = j
        dist[i0:i1] = np.sqrt(d2[np.arange(d2.shape[0]), j])

    return idx, dist


def pair_bead_centroids_mutual_nn(
    cyl_centroids: pd.DataFrame,
    nocyl_centroids: pd.DataFrame,
    *,
    max_pair_dist_px: float,
    pair_stem: str,
) -> pd.DataFrame:
    """
    Pair cylindrical/no-cylindrical bead centroids using mutual nearest neighbors.
    """
    if cyl_centroids.empty or nocyl_centroids.empty:
        return pd.DataFrame()

    cyl_xy = cyl_centroids[["xpix", "ypix"]].to_numpy(dtype=np.float64)
    nocyl_xy = nocyl_centroids[["xpix", "ypix"]].to_numpy(dtype=np.float64)

    c2n_idx, c2n_dist = _nearest_neighbor_indices(cyl_xy, nocyl_xy)
    n2c_idx, n2c_dist = _nearest_neighbor_indices(nocyl_xy, cyl_xy)

    rows: list[dict] = []

    for icyl, inocyl in enumerate(c2n_idx):
        if c2n_dist[icyl] > float(max_pair_dist_px):
            continue
        if n2c_idx[inocyl] != icyl:
            continue
        if n2c_dist[inocyl] > float(max_pair_dist_px):
            continue

        c = cyl_centroids.iloc[icyl]
        n = nocyl_centroids.iloc[inocyl]

        rows.append(
            {
                "pair_stem": pair_stem,
                "cyl_bead_id": int(c["bead_id"]),
                "nocyl_bead_id": int(n["bead_id"]),
                "x_cyl": float(c["xpix"]),
                "y_cyl": float(c["ypix"]),
                "x_nocyl": float(n["xpix"]),
                "y_nocyl": float(n["ypix"]),
                "raw_pair_distance_px": float(c2n_dist[icyl]),
                "cyl_n_detections": int(c["n_detections"]),
                "nocyl_n_detections": int(n["n_detections"]),
            }
        )

    return pd.DataFrame(rows)


def _rmse_xy(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def estimate_global_homography(
    correspondences: pd.DataFrame,
    settings: CylindricalLensCalibrationSettings,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Estimate the pooled global H_cyl_to_nocyl homography.
    """
    required = {"x_cyl", "y_cyl", "x_nocyl", "y_nocyl"}
    missing = required.difference(correspondences.columns)
    if missing:
        raise ValueError(f"Correspondence table missing columns: {sorted(missing)}")

    input_correspondences = correspondences.copy()

    src = input_correspondences[["x_cyl", "y_cyl"]].to_numpy(dtype=np.float64)
    dst = input_correspondences[["x_nocyl", "y_nocyl"]].to_numpy(dtype=np.float64)

    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    n_nonfinite_dropped = int(np.count_nonzero(~finite))

    if n_nonfinite_dropped:
        input_correspondences = input_correspondences.loc[finite].copy()
        src = src[finite]
        dst = dst[finite]

    if src.shape[0] < int(settings.min_pairs_for_homography):
        raise RuntimeError(
            f"Insufficient bead correspondences for homography: "
            f"need >= {settings.min_pairs_for_homography}, got {src.shape[0]}"
        )

    pre_rmse_px = _rmse_xy(src, dst)

    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(settings.ransac_reproj_thresh_px),
    )

    if H is None:
        raise RuntimeError("cv2.findHomography failed to estimate H_cyl_to_nocyl")

    inlier_mask = (
        mask.ravel().astype(bool)
        if mask is not None
        else np.ones(src.shape[0], dtype=bool)
    )

    pred = apply_homography(src, H)
    residual_xy = pred - dst
    residual_px = np.sqrt(np.sum(residual_xy**2, axis=1))

    corr = input_correspondences.copy()
    corr["x_cyl_transformed"] = pred[:, 0]
    corr["y_cyl_transformed"] = pred[:, 1]
    corr["dx_after_px"] = residual_xy[:, 0]
    corr["dy_after_px"] = residual_xy[:, 1]
    corr["residual_after_px"] = residual_px
    corr["ransac_inlier"] = inlier_mask

    post_rmse_all_px = _rmse_xy(pred, dst)
    post_rmse_inlier_px = _rmse_xy(pred[inlier_mask], dst[inlier_mask])

    summary = {
        "n_correspondences_input": int(len(correspondences)),
        "n_nonfinite_correspondences_dropped": int(n_nonfinite_dropped),
        "n_correspondences_total": int(src.shape[0]),
        "n_ransac_inliers": int(np.count_nonzero(inlier_mask)),
        "ransac_inlier_ratio": float(np.mean(inlier_mask)),
        "pre_rmse_px": float(pre_rmse_px),
        "post_rmse_all_px": float(post_rmse_all_px),
        "post_rmse_inlier_px": float(post_rmse_inlier_px),
        "median_residual_after_px": float(np.nanmedian(residual_px)),
        "max_residual_after_px": float(np.nanmax(residual_px)),
        "ransac_reproj_thresh_px": float(settings.ransac_reproj_thresh_px),
    }

    return H.astype(np.float64), corr, summary


def run_cylindrical_lens_homography_calibration(
    settings: CylindricalLensCalibrationSettings,
) -> dict:
    """
    Run the full folder-level cylindrical-lens homography calibration workflow.
    """
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_dir = output_dir / str(settings.diagnostics_dir_name)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    pair_outputs_root = diagnostics_dir / "pairs"
    pair_outputs_root.mkdir(parents=True, exist_ok=True)

    pairs = find_cylindrical_lens_movie_pairs(
        settings.calibration_dir,
        cylindrical_suffix=settings.cylindrical_suffix,
        no_cylindrical_suffix=settings.no_cylindrical_suffix,
        tiff_extension=settings.tiff_extension,
        input_search_mode=settings.input_search_mode,
    )

    logger.info(
        "Discovered %d cylindrical/no-cylindrical calibration pair(s) in %s",
        len(pairs),
        settings.calibration_dir,
    )

    channel_tag = _normalized_channel_tag(settings)

    dll = localizationlib(usecuda=int(settings.usecuda))

    pipeline_settings = settings.pipeline_settings_for_spline_fit()
    cylindrical_model = prepare_calibration_spline_channel_model(
        settings,
        pipeline_settings=pipeline_settings,
    )

    all_pair_tables: list[pd.DataFrame] = []
    pair_summaries: list[dict] = []

    for pair in pairs:
        pair_outdir = pair_outputs_root / pair.stem
        pair_outdir.mkdir(parents=True, exist_ok=True)

        cyl_csv = pair_outdir / f"{pair.stem}_cylindrical_spline_locs.csv"
        nocyl_csv = pair_outdir / f"{pair.stem}_no_cylindrical_gaussian_locs.csv"
        cyl_centroids_csv = pair_outdir / f"{pair.stem}_cylindrical_centroids.csv"
        nocyl_centroids_csv = pair_outdir / f"{pair.stem}_no_cylindrical_centroids.csv"
        bead_pairs_csv = pair_outdir / f"{pair.stem}_bead_pairs.csv"

        row = {
            "pair_stem": pair.stem,
            "cylindrical_tif": str(pair.cylindrical_tif),
            "no_cylindrical_tif": str(pair.no_cylindrical_tif),
            "status": "started",
            "error_message": "",
        }

        try:
            logger.info("========== Calibration pair: %s ==========", pair.stem)

            if settings.overwrite_existing_locs or not cyl_csv.exists():
                fit_stage1_single_channel(
                    channel_tag,
                    pair.cylindrical_tif,
                    cylindrical_model,
                    pipeline_settings,
                    dll,
                    out_csv=cyl_csv,
                    start_frame=0,
                    max_frames=settings.max_frames_per_movie,
                )
            else:
                logger.info("Reusing existing cylindrical spline locs: %s", cyl_csv)

            if settings.overwrite_existing_locs or not nocyl_csv.exists():
                fit_no_cylindrical_movie_gaussian(
                    pair.no_cylindrical_tif,
                    nocyl_csv,
                    settings,
                    dll,
                )
            else:
                logger.info("Reusing existing no-cylindrical Gaussian locs: %s", nocyl_csv)

            cyl_locs = pd.read_csv(cyl_csv)
            nocyl_locs = pd.read_csv(nocyl_csv)

            cyl_centroids = localizations_to_bead_centroids(
                cyl_locs,
                settings,
                source_label="cylindrical",
            )
            nocyl_centroids = localizations_to_bead_centroids(
                nocyl_locs,
                settings,
                source_label="no_cylindrical",
            )

            cyl_centroids.to_csv(cyl_centroids_csv, index=False)
            nocyl_centroids.to_csv(nocyl_centroids_csv, index=False)

            bead_pairs = pair_bead_centroids_mutual_nn(
                cyl_centroids,
                nocyl_centroids,
                max_pair_dist_px=float(settings.max_centroid_pair_dist_px),
                pair_stem=pair.stem,
            )
            bead_pairs.to_csv(bead_pairs_csv, index=False)

            if not bead_pairs.empty:
                all_pair_tables.append(bead_pairs)

            row.update(
                {
                    "status": "completed",
                    "cyl_locs_csv": str(cyl_csv),
                    "nocyl_locs_csv": str(nocyl_csv),
                    "cyl_locs_rows": int(len(cyl_locs)),
                    "nocyl_locs_rows": int(len(nocyl_locs)),
                    "cyl_centroids_csv": str(cyl_centroids_csv),
                    "nocyl_centroids_csv": str(nocyl_centroids_csv),
                    "cyl_centroids": int(len(cyl_centroids)),
                    "nocyl_centroids": int(len(nocyl_centroids)),
                    "bead_pairs_csv": str(bead_pairs_csv),
                    "bead_pairs": int(len(bead_pairs)),
                }
            )

        except Exception as e:
            logger.exception("Calibration pair %s failed.", pair.stem)
            row["status"] = "failed"
            row["error_message"] = str(e)

            if not settings.skip_failed_pairs:
                pair_summaries.append(row)
                pd.DataFrame(pair_summaries).to_csv(
                    diagnostics_dir / "calibration_pair_summary.csv",
                    index=False,
                )
                raise

        pair_summaries.append(row)
        pd.DataFrame(pair_summaries).to_csv(
            diagnostics_dir / "calibration_pair_summary.csv",
            index=False,
        )

    if not all_pair_tables:
        raise RuntimeError("No bead correspondences were generated from any calibration pair.")

    pooled = pd.concat(all_pair_tables, ignore_index=True)
    pooled_raw_csv = diagnostics_dir / "global_bead_correspondences_raw.csv"
    pooled.to_csv(pooled_raw_csv, index=False)

    H, pooled_with_fit, global_summary = estimate_global_homography(
        pooled,
        settings,
    )

    H_txt = output_dir / str(settings.homography_txt_name)
    H_npy = diagnostics_dir / str(settings.homography_npy_name)

    np.savetxt(H_txt, H, fmt="%.10f")
    np.save(H_npy, H)

    pooled_fit_csv = diagnostics_dir / "global_bead_correspondences_with_fit.csv"
    pooled_with_fit.to_csv(pooled_fit_csv, index=False)

    # Pair-level residual summary using the final global homography.
    pair_residual_summary = (
        pooled_with_fit.groupby("pair_stem", sort=True)["residual_after_px"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_correspondences",
                "mean": "mean_residual_after_px",
                "median": "median_residual_after_px",
                "max": "max_residual_after_px",
            }
        )
    )
    pair_residual_summary.to_csv(
        diagnostics_dir / "calibration_pair_residual_summary.csv",
        index=False,
    )

    global_summary.update(
        {
            "calibration_dir": str(settings.calibration_dir),
            "output_dir": str(output_dir),
            "diagnostics_dir": str(diagnostics_dir),
            "pair_outputs_dir": str(pair_outputs_root),
            "n_movie_pairs_discovered": int(len(pairs)),
            "n_movie_pairs_completed": int(
                sum(1 for r in pair_summaries if r.get("status") == "completed")
            ),
            "H_cyl_to_nocyl_txt": str(H_txt),
            "H_cyl_to_nocyl_npy": str(H_npy),
            "global_bead_correspondences_raw_csv": str(pooled_raw_csv),
            "global_bead_correspondences_with_fit_csv": str(pooled_fit_csv),
        }
    )

    homography_summary_csv = diagnostics_dir / "homography_summary.csv"
    pd.DataFrame([global_summary]).to_csv(homography_summary_csv, index=False)

    logger.info("Saved cylindrical-lens correction homography: %s", H_txt)
    logger.info("Saved homography summary: %s", homography_summary_csv)

    return {
        "H_cyl_to_nocyl": H,
        "H_npy": H_npy,
        "H_txt": H_txt,
        "homography_summary_csv": homography_summary_csv,
        "global_bead_correspondences_csv": pooled_fit_csv,
        "summary": global_summary,
    }