# workflow_orchestration.py

from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
from .loclib_ctypes import localizationlib
from .stage2_homography_registration import compute_projective_homography, RegParams

from .preprocessing_utils import (
    find_channel_file_pairs,
    photon_correct_single_frame,
    count_readable_tiff_prefix,
    ChannelFilePair,
)
from .roi_preparation import (
    display_preview_detections,
)
from .mle_fitting_drivers import (
    prepare_channel_models,
    fit_stage1_single_channel,
    fit_global_stage3_free_ratio,
    fit_global_stage3_fixed_ratios,
)
from .drift_correction_pipeline import (
    run_rcc_drift_correction,
    run_comet_drift_correction,
)

from .pipeline_config import (
    Settings,
    channel_split_channels_for_settings,
    psf_model_format_normalized,
)
from .pipeline_paths import (
    make_batch_output_root,
    make_intermediate_root,
    localization_registration_paths,
)
logger = logging.getLogger("stormpipe.workflow")

def _fit_mode_short_tag(settings: Settings) -> str:
    if settings.global_fit_mode == "free_ratio":
        return "free"
    if settings.global_fit_mode == "fixed_ratios":
        return "fixed"
    return str(settings.global_fit_mode)


def _applied_drift_tag(*, rcc_applied: bool, comet_applied: bool) -> str:
    if rcc_applied and comet_applied:
        return "rcc_comet"
    if rcc_applied:
        return "rcc"
    if comet_applied:
        return "comet"
    return "none"


def _count_csv_data_rows(path: str | Path | None) -> int | None:
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        return None

    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        logger.warning("Could not count CSV rows: %s", p, exc_info=True)
        return None


def _record_existing_csv(
    summary: dict,
    *,
    path_key: str,
    rows_key: str,
    path: Path,
) -> None:
    if path.exists():
        summary[path_key] = str(path)
        summary[rows_key] = _count_csv_data_rows(path)
    else:
        summary[path_key] = ""
        summary[rows_key] = None


def _record_channel_split_outputs(
    summary: dict,
    *,
    channel_split_output_dir: Path,
    output_stem: str,
    channels: tuple[int, ...],
) -> None:
    for ch in channels:
        p = Path(channel_split_output_dir) / f"{output_stem}_ch{int(ch)}.csv"
        key_prefix = f"render_ch{int(ch)}"
        if p.exists():
            summary[f"{key_prefix}_csv"] = str(p)
            summary[f"{key_prefix}_rows"] = _count_csv_data_rows(p)
        else:
            summary[f"{key_prefix}_csv"] = ""
            summary[f"{key_prefix}_rows"] = None


_COMPACT_RUN_SUMMARY_BASE_COLUMNS = [
    "pair_stem",
    "psf_model_format",
    "pipeline_mode",
    "global_fit_mode",
    "stage3_full_output_locs",
]

_STAGE1_SUMMARY_COLUMNS = [
    "pair_stem",
    "psf_model_format",
    "R_stage1_output_locs",
    "T_stage1_output_locs",
]


def _stage1_summary_row_for_pair(
    *,
    settings: Settings,
    pair: ChannelFilePair,
    intermediate_dir: Path,
) -> dict:
    """
    Build one row for the Stage 1 intermediate summary.

    This summary is written only for full pipeline runs and lives in:
        data_dir / settings.intermediate_dir_name / stage1_summary.csv
    """
    paths = localization_registration_paths(intermediate_dir, pair.stem)

    return {
        "pair_stem": pair.stem,
        "psf_model_format": psf_model_format_normalized(settings),
        "R_stage1_output_locs": _count_csv_data_rows(paths["R_stage1_csv"]),
        "T_stage1_output_locs": _count_csv_data_rows(paths["T_stage1_csv"]),
    }


def _write_stage1_summary(
    intermediate_dir: Path,
    rows: list[dict],
) -> None:
    """
    Write the compact Stage 1 summary CSV.

    The file is intentionally overwritten during full pipeline runs so it reflects
    the current set of Stage 1 intermediates produced by that run.
    """
    intermediate_dir = Path(intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    stage1_summary_csv = intermediate_dir / "stage1_summary.csv"
    pd.DataFrame(rows, columns=_STAGE1_SUMMARY_COLUMNS).to_csv(
        stage1_summary_csv,
        index=False,
    )
    logger.info("Saved Stage 1 summary CSV: %s", stage1_summary_csv)

def _channel_render_locs_sort_key(col: str) -> int:
    """
    Sort compact channel output columns numerically:

        ch1_TS_render_ready_output_locs
        ch2_TS_render_ready_output_locs
        ...
    """
    try:
        return int(str(col).split("_", 1)[0].replace("ch", ""))
    except Exception:
        return 10_000


def _add_compact_run_summary_fields(summary: dict, settings: Settings) -> dict:
    """
    Add the small user-facing run_summary.csv fields.

    The verbose internal summary keys are kept for bookkeeping while processing,
    but _write_run_summary writes only these compact fields.
    """
    summary["psf_model_format"] = psf_model_format_normalized(settings)

    # Rename stage3_full_rows to the user-facing meaning.
    summary["stage3_full_output_locs"] = summary.get("stage3_full_rows")

    channels = channel_split_channels_for_settings(settings)

    if settings.global_fit_mode == "free_ratio":
        # Free-ratio mode has one render-ready output channel, normally ch0.
        ch = int(channels[0]) if channels else int(settings.free_ratio_output_channel)
        summary["TS_render_ready_output_locs"] = summary.get(f"render_ch{ch}_rows")

    elif settings.global_fit_mode == "fixed_ratios":
        # Fixed-ratio mode exports one render-ready output per assigned channel.
        for ch in channels:
            ch = int(ch)
            summary[f"ch{ch}_TS_render_ready_output_locs"] = summary.get(
                f"render_ch{ch}_rows"
            )

    return summary


def _compact_run_summary_dataframe(summaries: list[dict]) -> pd.DataFrame:
    """
    Convert verbose internal per-pair summaries into the compact run_summary.csv
    schema.
    """
    df = pd.DataFrame(summaries)

    if df.empty:
        return df

    out = pd.DataFrame(index=df.index)

    for col in _COMPACT_RUN_SUMMARY_BASE_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = None

    # Free-ratio render-ready localization count.
    if "TS_render_ready_output_locs" in df.columns:
        out["TS_render_ready_output_locs"] = df["TS_render_ready_output_locs"]

    # Fixed-ratio render-ready localization counts.
    channel_cols = sorted(
        [
            c for c in df.columns
            if str(c).startswith("ch")
            and str(c).endswith("_TS_render_ready_output_locs")
        ],
        key=_channel_render_locs_sort_key,
    )

    for col in channel_cols:
        out[col] = df[col]

    return out


def _write_run_summary(batch_outdir: Path, summaries: list[dict]) -> None:
    if not summaries:
        return

    batch_outdir = Path(batch_outdir)
    batch_outdir.mkdir(parents=True, exist_ok=True)

    summary_csv = batch_outdir / "run_summary.csv"
    _compact_run_summary_dataframe(summaries).to_csv(summary_csv, index=False)
    logger.info("Saved compact run summary CSV: %s", summary_csv)

def preview_channel_pair_detections(
    *,
    settings: Settings,
    pair: ChannelFilePair,
) -> dict:
    """
    Display detection ROIs for one reflected/transmitted TIFF pair without writing
    pipeline outputs.
    """
    if settings.pipeline_mode == "globloc_only":
        message = (
            "Peak-detection preview is only available in pipeline_mode='full'. "
            "In pipeline_mode='globloc_only', the workflow reuses existing Stage 1 "
            "spline-localization CSVs and does not run raw peak detection."
        )
        logger.info("[%s] %s", pair.stem, message)
        print(message)
        return {
            "pair_stem": pair.stem,
            "R_tif": str(pair.tif_R),
            "T_tif": str(pair.tif_T),
            "pipeline_mode": settings.pipeline_mode,
            "status": "preview_not_available",
            "message": message,
            "error_message": "",
        } 
    logger.info("========== Previewing pair: %s ==========", pair.stem)
    logger.info("R tif: %s", pair.tif_R)
    logger.info("T tif: %s", pair.tif_T)

    nR_reported, nR = count_readable_tiff_prefix(pair.tif_R)
    nT_reported, nT = count_readable_tiff_prefix(pair.tif_T)

    logger.info(
        "[%s] Channel R frames: reported=%d readable=%d",
        pair.stem,
        nR_reported,
        nR,
    )
    logger.info(
        "[%s] Channel T frames: reported=%d readable=%d",
        pair.stem,
        nT_reported,
        nT,
    )

    n_min = min(nR, nT)
    if not (0 <= settings.preview_frame_index < n_min):
        raise ValueError(
            f"PREVIEW_DETECTIONS_FRAME={settings.preview_frame_index} "
            f"exceeds readable frame range for pair {pair.stem!r} (0–{n_min-1})"
        )

    fr = photon_correct_single_frame(
        pair.tif_R,
        settings.preview_frame_index,
        offset=settings.offset,
        camera_gain=settings.camera_gain,
        QE=settings.qe,
    )
    ft = photon_correct_single_frame(
        pair.tif_T,
        settings.preview_frame_index,
        offset=settings.offset,
        camera_gain=settings.camera_gain,
        QE=settings.qe,
    )

    nRpk, nTpk, shown_f = display_preview_detections(
        fr[None, ...],
        ft[None, ...],
        0,
        roi_size=settings.roi_size,
        thr_R=settings.thr_factor_reflected,
        thr_T=settings.thr_factor_transmitted,
        s1=settings.sigma1,
        s2=settings.sigma2,
        keep_peak_winner=settings.keep_peak_winner,
        mindist=settings.min_distance,
        display_index=settings.preview_frame_index,
    )

    logger.info(
        "[%s] Preview: frame %s | R peaks=%d, T peaks=%d",
        pair.stem,
        shown_f,
        nRpk,
        nTpk,
    )

    plt.show(block=True)
    plt.close("all")

    return {
        "pair_stem": pair.stem,
        "R_tif": str(pair.tif_R),
        "T_tif": str(pair.tif_T),
        "nR_reported": int(nR_reported),
        "nR_readable": int(nR),
        "nT_reported": int(nT_reported),
        "nT_readable": int(nT),
        "preview_frame_index": int(settings.preview_frame_index),
        "preview_frame_shown": int(shown_f),
        "preview_R_peaks": int(nRpk),
        "preview_T_peaks": int(nTpk),
        "pipeline_mode": settings.pipeline_mode,
        "global_fit_mode": settings.global_fit_mode,
        "status": "preview_only",
        "error_message": "",
    }

def preview_batch_detections(settings: Settings) -> list[dict]:
    """
    Display raw peak-detection previews for all matched reflected/transmitted TIFF
    pairs without creating pipeline output folders.
    """
    if settings.pipeline_mode == "globloc_only":
        message = (
            "Peak-detection preview is only available in pipeline_mode='full'. "
            "In pipeline_mode='globloc_only', the workflow reuses existing Stage 1 "
            "spline-localization CSVs and does not run raw peak detection."
        )
        logger.info(message)
        print(message)
        return [{
            "pipeline_mode": settings.pipeline_mode,
            "status": "preview_not_available",
            "message": message,
        }]

    experiment_dir = Path(settings.data_dir)
    pairs = find_channel_file_pairs(
        experiment_dir,
        reflected_suffix=settings.reflected_suffix,
        transmitted_suffix=settings.transmitted_suffix,
        tiff_extension=settings.tiff_extension,
        input_search_mode=settings.input_search_mode,
        min_readable_frames_per_channel=settings.min_readable_frames_per_channel,
    )

    logger.info("Discovered %d R/T pairs in %s", len(pairs), experiment_dir)

    summaries: list[dict] = []

    for pair in pairs:
        try:
            row = preview_channel_pair_detections(settings=settings, pair=pair)
        except Exception as e:
            logger.exception("Preview for pair %s failed. Continuing batch.", pair.stem)
            row = {
                "pair_stem": pair.stem,
                "R_tif": str(pair.tif_R),
                "T_tif": str(pair.tif_T),
                "pipeline_mode": settings.pipeline_mode,
                "global_fit_mode": settings.global_fit_mode,
                "status": "failed",
                "error_message": str(e),
            }

        summaries.append(row)

    logger.info("Preview complete for %d R/T pair(s).", len(summaries))
    return summaries

#workflow execution
def process_channel_pair(
    *,
    settings: Settings,
    pair: ChannelFilePair,
    pair_output_dir: Path,
    channel_split_output_dir: Path,
    chan_models: dict[str, object] | None,
    dll,
) -> dict:
    """
    Run one R/T pair.

    Full mode:
        run Stage 1 + Stage 2 + Stage 3 + optional drift correction.

    GlobLoc-only mode:
        reuse existing Stage 1/2 intermediates and run Stage 3 + optional drift correction.
    """
    if settings.preview_detections:
        return preview_channel_pair_detections(settings=settings, pair=pair)

    pair_output_dir = Path(pair_output_dir)
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    fit_tag = _fit_mode_short_tag(settings)

    experiment_dir = Path(settings.data_dir)
    intermediate_dir = make_intermediate_root(settings)
    paths = localization_registration_paths(intermediate_dir, pair.stem)

    logger.info("========== Processing pair: %s ==========", pair.stem)
    logger.info("R tif: %s", pair.tif_R)
    logger.info("T tif: %s", pair.tif_T)
    logger.info("Pair Stage 3 output dir: %s", pair_output_dir)
    logger.info("Pair Stage 1/2 intermediate dir: %s", intermediate_dir)

    nR_reported, nR = count_readable_tiff_prefix(pair.tif_R)
    nT_reported, nT = count_readable_tiff_prefix(pair.tif_T)

    logger.info(
        "[%s] Channel R frames: reported=%d readable=%d",
        pair.stem,
        nR_reported,
        nR,
    )
    logger.info(
        "[%s] Channel T frames: reported=%d readable=%d",
        pair.stem,
        nT_reported,
        nT,
    )

    summary = {
        "pair_stem": pair.stem,
        "R_tif": str(pair.tif_R),
        "T_tif": str(pair.tif_T),
        "pair_output_dir": str(pair_output_dir),
        "nR_reported": int(nR_reported),
        "nR_readable": int(nR),
        "nT_reported": int(nT_reported),
        "nT_readable": int(nT),
        "psf_model_format": psf_model_format_normalized(settings),
        "pipeline_mode": settings.pipeline_mode,
        "global_fit_mode": settings.global_fit_mode,
        "status": "started",
        "error_message": "",
    }


    # -------------------------------------------------------------------------
    # Stage 1 + Stage 2 intermediates
    # -------------------------------------------------------------------------
    if settings.pipeline_mode == "full":
        if chan_models is None:
            raise RuntimeError("chan_models must be provided when pipeline_mode='full'")

        fit_stage1_single_channel(
            "R",
            pair.tif_R,
            chan_models["R"],
            settings,
            dll,
            out_csv=paths["R_stage1_csv"],
            start_frame=0,
            max_frames=nR,
        )

        fit_stage1_single_channel(
            "T",
            pair.tif_T,
            chan_models["T"],
            settings,
            dll,
            out_csv=paths["T_stage1_csv"],
            start_frame=0,
            max_frames=nT,
        )

        logger.info("[%s] Stage 1 complete.", pair.stem)

        registration_frame_start = int(settings.registration_start_frame)
        registration_frame_end = settings.registration_end_frame

        if registration_frame_end is not None:
            registration_frame_end = int(registration_frame_end)

        n_registration_frames = min(int(nR), int(nT))
        registration_window_valid = (
            registration_frame_end is not None
            and 0 <= registration_frame_start < registration_frame_end <= n_registration_frames
        )

        if not registration_window_valid:
            logger.warning(
                "[%s] Requested registration frame window [%s, %s) is outside the "
                "readable shared frame range [0, %d). Using all frames for homography.",
                pair.stem,
                registration_frame_start,
                registration_frame_end,
                n_registration_frames,
            )
            registration_frame_start_for_params = None
            registration_frame_end_for_params = None
        else:
            logger.info(
                "[%s] Estimating Stage 2 homography using frames [%d, %d).",
                pair.stem,
                registration_frame_start,
                registration_frame_end,
            )
            registration_frame_start_for_params = registration_frame_start
            registration_frame_end_for_params = registration_frame_end

        reg_params = RegParams(
            max_pairing_dist_px=4.0,
            ransac_reproj_thresh_px=2.0,
            min_photons=None,
            max_locprec_nm=None,
            frame_start=registration_frame_start_for_params,
            frame_end=registration_frame_end_for_params,
            write_transformed_T_csv=True,
            transformed_T_csv_name=paths["T_stage1_in_R_csv"].name,
            homography_npy_name=paths["homography_npy"].name,
            homography_txt_name=paths["homography_txt"].name,
        )

        diags = compute_projective_homography(
            outdir=intermediate_dir,
            params=reg_params,
            csv_R=paths["R_stage1_csv"],
            csv_T=paths["T_stage1_csv"],
            pixelsize_nm=settings.pixelsize_nm,
        )

        H_input = diags["H"]

        summary.update({
            "registration_pairs": int(diags["total_pairs"]),
            "registration_inliers": int(diags["inliers"]),
            "registration_inlier_ratio": float(diags["inlier_ratio"]),
            "registration_pre_rmse_px": float(diags["pre_rmse_px"]),
            "registration_post_rmse_px": float(diags["post_rmse_px"]),
            "registration_used_requested_frame_window": bool(
                diags.get("registration_used_requested_frame_window", False)
            ),
            "registration_frame_start": diags.get("registration_frame_start"),
            "registration_frame_end": diags.get("registration_frame_end"),
            "homography_txt": str(paths["homography_txt"]),
            "homography_npy": str(paths["homography_npy"]),
        })

        logger.info(
            "[%s] Stage 2 registration complete: pairs=%d inliers=%d "
            "inlier_ratio=%.2f preRMSE=%.3fpx postRMSE=%.3fpx",
            pair.stem,
            diags["total_pairs"],
            diags["inliers"],
            diags["inlier_ratio"],
            diags["pre_rmse_px"],
            diags["post_rmse_px"],
        )

    elif settings.pipeline_mode == "globloc_only":
        required = [
            paths["R_stage1_csv"],
            paths["T_stage1_csv"],
            paths["homography_txt"],
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot run globloc_only mode for "
                f"{pair.stem!r}; missing intermediate file(s): "
                + ", ".join(str(p) for p in missing)
            )

        H_input = paths["homography_txt"]

        logger.info(
            "[%s] GlobLoc-only mode: reusing Stage 1/2 intermediates from %s",
            pair.stem,
            intermediate_dir,
        )

    else:
        raise ValueError(f"Unsupported pipeline_mode={settings.pipeline_mode!r}")

    if not settings.run_stage3_global_fit:
        logger.info("[%s] Stage 3 disabled; stopping after Stage 1/2.", pair.stem)
        summary["stage3_csv"] = ""
        summary["rcc_csv"] = ""
        summary["status"] = "stage1_stage2_only"
        _add_compact_run_summary_fields(summary, settings)
        return summary
    
    # -------------------------------------------------------------------------
    # Stage 3 global fit
    # -------------------------------------------------------------------------
    logger.info(
        "[%s] Beginning Stage 3 global dual-channel fit (%s).",
        pair.stem,
        settings.global_fit_mode,
    )

    if settings.global_fit_mode == "free_ratio":
        global_csv = fit_global_stage3_free_ratio(
            pair.tif_R,
            pair.tif_T,
            H_input,
            settings,
            dll,
            outdir=pair_output_dir,
            R_stage1_csv=paths["R_stage1_csv"],
            T_stage1_csv=paths["T_stage1_csv"],
            start_frame=settings.stage3_start_frame,
            max_frames=settings.stage3_max_frames,
            pair_stem=pair.stem,
        )

    elif settings.global_fit_mode == "fixed_ratios":
        global_csv = fit_global_stage3_fixed_ratios(
            pair.tif_R,
            pair.tif_T,
            H_input,
            settings,
            dll,
            outdir=pair_output_dir,
            R_stage1_csv=paths["R_stage1_csv"],
            T_stage1_csv=paths["T_stage1_csv"],
            start_frame=settings.stage3_start_frame,
            max_frames=settings.stage3_max_frames,
            pair_stem=pair.stem,
        )

    else:
        raise ValueError(f"Unsupported global_fit_mode={settings.global_fit_mode!r}")

    logger.info("[%s] Stage 3 complete: %s", pair.stem, global_csv)

    stage3_full_csv = pair_output_dir / f"{pair.stem}_locs_{fit_tag}_full.csv"
    stage3_filtered_csv = pair_output_dir / f"{pair.stem}_locs_{fit_tag}_filtered.csv"

    summary["stage3_csv"] = str(global_csv)
    _record_existing_csv(
        summary,
        path_key="stage3_full_csv",
        rows_key="stage3_full_rows",
        path=stage3_full_csv,
    )
    _record_existing_csv(
        summary,
        path_key="stage3_filtered_csv",
        rows_key="stage3_filtered_rows",
        path=stage3_filtered_csv,
    )

    # -------------------------------------------------------------------------
    # Drift correction + final channel split
    #
    # Rules:
    #   - RCC only:        Stage 3 -> RCC
    #   - COMET only:      Stage 3 -> COMET
    #   - RCC + COMET:     Stage 3 -> RCC -> COMET
    #
    # Only the final enabled correction stage writes channel-split CSVs.
    # -------------------------------------------------------------------------
    final_corrected_csv = Path(global_csv)

    summary["rcc_csv"] = ""
    summary["comet_csv"] = ""
    summary["final_drift_corrected_csv"] = str(final_corrected_csv)

    if settings.run_rcc_drift_correction and settings.run_comet_drift_correction:
        summary["drift_correction_mode"] = "rcc_then_comet"
    elif settings.run_rcc_drift_correction:
        summary["drift_correction_mode"] = "rcc"
    elif settings.run_comet_drift_correction:
        summary["drift_correction_mode"] = "comet"
    else:
        summary["drift_correction_mode"] = "none"

    summary["rcc_applied"] = False
    summary["comet_applied"] = False

    if settings.run_rcc_drift_correction:
        logger.info("[%s] Running RCC drift correction.", pair.stem)

        rcc_output_file = pair_output_dir / f"{pair.stem}_locs_{fit_tag}_rcc_drift.csv"
        rcc_drift_table_file = pair_output_dir / f"{pair.stem}_drift_rcc.csv"
        rcc_channel_stem = f"{pair.stem}_{fit_tag}_rcc"

        before_rcc_csv = Path(final_corrected_csv)

        final_corrected_csv = run_rcc_drift_correction(
            final_corrected_csv,
            output_file=rcc_output_file,
            drift_table_file=rcc_drift_table_file,
            frame_col="frame",
            x_col="x [nm]",
            y_col="y [nm]",
            z_col="z [nm]",
            correct_xy=settings.rcc_correct_xy,
            correct_z=settings.rcc_correct_z,
            xy_timepoints=settings.rcc_xy_timepoints,
            xy_pixel_size_nm=settings.rcc_xy_pixel_size_nm,
            z_timepoints=settings.rcc_z_timepoints,
            z_bin_width_nm=settings.rcc_z_bin_width_nm,
            max_drift_nm=settings.rcc_max_drift_nm,
            z_range_nm=settings.rcc_z_range_nm,
            save_channel_split_csvs=not settings.run_comet_drift_correction,
            channel_split_channels=channel_split_channels_for_settings(settings),
            channel_split_output_dir=channel_split_output_dir,
            channel_split_output_stem=rcc_channel_stem,
            render_ready_channel_csvs=settings.render_ready_channel_csvs,
            render_max_number_in_group=settings.render_max_number_in_group,
            combine_grouped_localizations_for_render=(
                settings.combine_grouped_localizations_for_render
            ),
            render_frame_offset=settings.render_frame_offset,
            render_z_sign=settings.render_z_sign,
            render_z_scale=settings.render_z_scale,
            render_uncertainty_xy_scale=settings.render_uncertainty_xy_scale,
            render_uncertainty_z_scale=settings.render_uncertainty_z_scale,
            apply_cylindrical_lens_xy_correction_for_render=(
                settings.apply_cylindrical_lens_xy_correction_for_render
            ),
            cylindrical_lens_xy_homography=settings.cylindrical_lens_xy_homography,
            pixelsize_nm=settings.pixelsize_nm,
            skip_on_failure=settings.rcc_skip_on_failure,
            overwrite=True,
        )

        rcc_applied = (
            Path(final_corrected_csv).resolve(strict=False)
            != before_rcc_csv.resolve(strict=False)
        )
        summary["rcc_applied"] = bool(rcc_applied)
        summary["rcc_csv"] = str(final_corrected_csv)
        summary["rcc_drift_csv"] = str(rcc_drift_table_file) if rcc_drift_table_file.exists() else ""

        logger.info("[%s] RCC drift correction complete: %s", pair.stem, final_corrected_csv)

        if rcc_applied and not settings.run_comet_drift_correction:
            _record_channel_split_outputs(
                summary,
                channel_split_output_dir=channel_split_output_dir,
                output_stem=rcc_channel_stem,
                channels=channel_split_channels_for_settings(settings),
            )

    if settings.run_comet_drift_correction:
        logger.info("[%s] Running COMET drift correction.", pair.stem)

        comet_tag = "rcc_comet" if bool(summary.get("rcc_applied", False)) else "comet"
        comet_output_file = pair_output_dir / f"{pair.stem}_locs_{fit_tag}_{comet_tag}_drift.csv"
        comet_drift_table_file = pair_output_dir / f"{pair.stem}_drift_comet.csv"
        comet_channel_stem = f"{pair.stem}_{fit_tag}_{comet_tag}"

        before_comet_csv = Path(final_corrected_csv)

        final_corrected_csv = run_comet_drift_correction(
            final_corrected_csv,
            output_file=comet_output_file,
            drift_table_file=comet_drift_table_file,
            frame_col="frame",
            x_col="x [nm]",
            y_col="y [nm]",
            z_col="z [nm]",
            segmentation_mode=settings.comet_segmentation_mode,
            segmentation_var=settings.comet_segmentation_var,
            initial_sigma_nm=settings.comet_initial_sigma_nm,
            target_sigma_nm=settings.comet_target_sigma_nm,
            max_drift_nm=settings.comet_max_drift_nm,
            boxcar_width=settings.comet_boxcar_width,
            interpolation_method=settings.comet_interpolation_method,
            max_locs_per_segment=settings.comet_max_locs_per_segment,
            suppress_comet_output=settings.comet_suppress_output,
            save_channel_split_csvs=True,
            channel_split_channels=channel_split_channels_for_settings(settings),
            channel_split_output_dir=channel_split_output_dir,
            channel_split_output_stem=comet_channel_stem,
            render_ready_channel_csvs=settings.render_ready_channel_csvs,
            render_max_number_in_group=settings.render_max_number_in_group,
            combine_grouped_localizations_for_render=(
                settings.combine_grouped_localizations_for_render
            ),
            render_frame_offset=settings.render_frame_offset,
            render_z_sign=settings.render_z_sign,
            render_z_scale=settings.render_z_scale,
            render_uncertainty_xy_scale=settings.render_uncertainty_xy_scale,
            render_uncertainty_z_scale=settings.render_uncertainty_z_scale,
            apply_cylindrical_lens_xy_correction_for_render=(
                settings.apply_cylindrical_lens_xy_correction_for_render
            ),
            cylindrical_lens_xy_homography=settings.cylindrical_lens_xy_homography,
            pixelsize_nm=settings.pixelsize_nm,
            skip_on_failure=settings.comet_skip_on_failure,
            overwrite=True,
        )

        comet_applied = (
            Path(final_corrected_csv).resolve(strict=False)
            != before_comet_csv.resolve(strict=False)
        )
        summary["comet_applied"] = bool(comet_applied)
        summary["comet_csv"] = str(final_corrected_csv)
        summary["comet_drift_csv"] = (
            str(comet_drift_table_file) if comet_drift_table_file.exists() else ""
        )

        logger.info("[%s] COMET drift correction complete: %s", pair.stem, final_corrected_csv)

        if comet_applied:
            _record_channel_split_outputs(
                summary,
                channel_split_output_dir=channel_split_output_dir,
                output_stem=comet_channel_stem,
                channels=channel_split_channels_for_settings(settings),
            )

    summary["final_drift_tag"] = _applied_drift_tag(
        rcc_applied=bool(summary.get("rcc_applied", False)),
        comet_applied=bool(summary.get("comet_applied", False)),
    )
    summary["final_drift_corrected_csv"] = str(final_corrected_csv)
    summary["final_drift_corrected_rows"] = _count_csv_data_rows(final_corrected_csv)
    summary["status"] = "completed"

    _add_compact_run_summary_fields(summary, settings)
    return summary


def run_batch(settings: Settings) -> None:
    if settings.preview_detections:
        preview_batch_detections(settings)
        return

    experiment_dir = Path(settings.data_dir)
    pairs = find_channel_file_pairs(
        experiment_dir,
        reflected_suffix=settings.reflected_suffix,
        transmitted_suffix=settings.transmitted_suffix,
        tiff_extension=settings.tiff_extension,
        input_search_mode=settings.input_search_mode,
        min_readable_frames_per_channel=settings.min_readable_frames_per_channel,
    )

    logger.info("Discovered %d R/T pairs in %s", len(pairs), experiment_dir)

    batch_outdir = make_batch_output_root(settings)
    intermediate_dir = make_intermediate_root(settings)

    channel_split_output_dir = batch_outdir / "thunderstorm_render_ready_channels"
    channel_split_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Batch GlobLoc output folder: %s", batch_outdir)
    logger.info("Reusable Stage 1/2 intermediate folder: %s", intermediate_dir)
    logger.info("ThunderSTORM render-ready channel CSV folder: %s", channel_split_output_dir)

    dll = localizationlib(usecuda=1)

    chan_models = None
    if settings.pipeline_mode == "full":
        chan_models = prepare_channel_models(settings)

    summaries: list[dict] = []

    stage1_summaries: list[dict] = []
    write_stage1_summary = settings.pipeline_mode == "full"

    if write_stage1_summary:
        # Start each full run with a fresh Stage 1 summary file.
        _write_stage1_summary(intermediate_dir, stage1_summaries)

    for pair in pairs:
        pair_output_dir = batch_outdir / pair.stem

        try:
            row = process_channel_pair(
                settings=settings,
                pair=pair,
                pair_output_dir=pair_output_dir,
                channel_split_output_dir=channel_split_output_dir,
                chan_models=chan_models,
                dll=dll,
            )
        except Exception as e:
            logger.exception("Pair %s failed. Continuing batch.", pair.stem)
            row = {
                "pair_stem": pair.stem,
                "R_tif": str(pair.tif_R),
                "T_tif": str(pair.tif_T),
                "pair_output_dir": str(pair_output_dir),
                "psf_model_format": psf_model_format_normalized(settings),
                "pipeline_mode": settings.pipeline_mode,
                "global_fit_mode": settings.global_fit_mode,
                "status": "failed",
                "error_message": str(e),
            }
            _add_compact_run_summary_fields(row, settings)

        summaries.append(row)
        _write_run_summary(batch_outdir, summaries)

        if write_stage1_summary:
            stage1_summaries.append(
                _stage1_summary_row_for_pair(
                    settings=settings,
                    pair=pair,
                    intermediate_dir=intermediate_dir,
                )
            )
            _write_stage1_summary(intermediate_dir, stage1_summaries)

    logger.info("Batch complete. Output folder: %s", batch_outdir)


def run(settings: Settings) -> None:
    run_batch(settings)