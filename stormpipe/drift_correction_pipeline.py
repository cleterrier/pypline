# drift_correction_pipeline.py
from __future__ import annotations

from pathlib import Path
import logging
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

from .rcc_drift import RCCDriftConfig, correct_drift_xyz
from .thunderstorm_render_ready_exports import to_render_ready_dataframe

logger = logging.getLogger("stormpipe.drift_correction")

def load_xy_homography(homography_path: str | Path) -> np.ndarray:
    """
    Load a 3x3 xy homography from .npy or text format.
    """
    p = Path(homography_path)

    if not p.exists():
        raise FileNotFoundError(f"XY homography file not found: {p}")

    if p.suffix.lower() == ".npy":
        H = np.load(p).astype(np.float64)
    else:
        H = np.loadtxt(p).astype(np.float64)

    if H.shape != (3, 3):
        raise ValueError(f"XY homography must have shape (3, 3), got {H.shape}")

    return H


def apply_xy_homography_to_localization_dataframe(
    locs: pd.DataFrame,
    homography_path: str | Path,
    *,
    pixelsize_nm: float,
    xpix_col: str = "xpix",
    ypix_col: str = "ypix",
    x_nm_col: str = "x [nm]",
    y_nm_col: str = "y [nm]",
) -> pd.DataFrame:
    """
    Apply an xy homography to localization coordinates in pixel space.

    The homography is applied to xpix/ypix. The nm coordinate columns are then
    recomputed using the pipeline convention:

        x [nm] = (xpix + 1) * pixelsize_nm
        y [nm] = (ypix + 1) * pixelsize_nm

    (This is specifically for correcting the cylindrical lens distortion in the xy plane
    for the thunderstorm render-ready exports. The homography should be pre-calibrated)
    """
    if locs is None or locs.empty:
        return locs.copy()

    missing = [c for c in (xpix_col, ypix_col) if c not in locs.columns]
    if missing:
        raise ValueError(
            "Cannot apply cylindrical-lens xy homography. Missing column(s): "
            + ", ".join(missing)
        )

    H = load_xy_homography(homography_path)

    out = locs.copy()

    x = pd.to_numeric(out[xpix_col], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out[ypix_col], errors="coerce").to_numpy(dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)

    if not np.any(finite):
        logger.warning(
            "No finite xpix/ypix values found while applying xy homography."
        )
        return out

    pts = np.column_stack([x[finite], y[finite], np.ones(np.count_nonzero(finite))])
    transformed = (H @ pts.T).T

    denom = transformed[:, 2]
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)

    x_new = np.full(np.count_nonzero(finite), np.nan, dtype=np.float64)
    y_new = np.full(np.count_nonzero(finite), np.nan, dtype=np.float64)

    x_new[valid] = transformed[valid, 0] / denom[valid]
    y_new[valid] = transformed[valid, 1] / denom[valid]

    finite_indices = np.flatnonzero(finite)

    out.loc[out.index[finite_indices], xpix_col] = x_new
    out.loc[out.index[finite_indices], ypix_col] = y_new

    out[x_nm_col] = (pd.to_numeric(out[xpix_col], errors="coerce") + 1.0) * float(pixelsize_nm)
    out[y_nm_col] = (pd.to_numeric(out[ypix_col], errors="coerce") + 1.0) * float(pixelsize_nm)

    logger.info(
        "Applied cylindrical-lens xy homography to %d localization(s): %s",
        int(np.count_nonzero(finite)),
        homography_path,
    )

    return out

def write_channel_split_csvs(
    locs: pd.DataFrame,
    source_csv_path: str | Path,
    *,
    channel_col: str = "channel",
    channels: tuple[int, ...] = (0, 1, 2, 3, 4),
    output_dir: str | Path | None = None,
    output_stem: str | None = None,
    overwrite: bool = True,
    render_ready: bool = True,
    render_max_number_in_group: int | None = 5,
    combine_grouped_localizations_for_render: bool = True,
    render_frame_offset: int = 1,
    render_z_sign: float = 1.0,
    render_z_scale: float = 1.0,
    render_uncertainty_xy_scale: float = 1.0,
    render_uncertainty_z_scale: float = 1.0,
    apply_cylindrical_lens_xy_correction_for_render: bool = False,
    cylindrical_lens_xy_homography: str | Path | None = None,
    pixelsize_nm: float | None = None,
) -> dict[int, Path]:
    """
    Write one CSV per requested localization channel.

    """
    source_csv_path = Path(source_csv_path)

    if output_dir is None:
        output_dir = source_csv_path.with_name(source_csv_path.stem + "_channels")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if locs is None or locs.empty:
        logger.warning("No localizations available for channel split: %s", source_csv_path)
        return {}

    if channel_col not in locs.columns:
        logger.warning(
            "Cannot write channel-split CSVs because column %r is missing from %s",
            channel_col,
            source_csv_path,
        )
        return {}

    channel_values = pd.to_numeric(locs[channel_col], errors="coerce")
    written: dict[int, Path] = {}

    for ch in channels:
        mask = channel_values == int(ch)
        df_ch = locs.loc[mask].copy()

        if df_ch.empty:
            logger.warning(
                "No localizations found for channel %d in %s; no channel file written.",
                int(ch),
                source_csv_path,
            )
            continue

        file_stem = source_csv_path.stem if output_stem is None else str(output_stem)
        out_csv = output_dir / f"{file_stem}_ch{int(ch)}.csv"

        if out_csv.exists() and not overwrite:
            logger.info("Channel CSV already exists, skipping: %s", out_csv)
            written[int(ch)] = out_csv
            continue

        df_export = df_ch

        if render_ready and render_max_number_in_group is not None:
            from .stage3_filtering_grouping import filter_groups_by_size

            before_group_filter = len(df_export)
            df_export = filter_groups_by_size(
                df_export,
                max_number_in_group=int(render_max_number_in_group),
            )
            logger.info(
                "Applied render max-group filter | channel=%d max_group=%d "
                "rows_before=%d rows_after=%d",
                int(ch),
                int(render_max_number_in_group),
                int(before_group_filter),
                int(len(df_export)),
            )

        if (
            render_ready
            and combine_grouped_localizations_for_render
            and "groupindex" in df_export.columns
        ):
            from .stage3_filtering_grouping import combine_grouped_localizations

            before_combine = len(df_export)
            df_export = combine_grouped_localizations(df_export)
            logger.info(
                "Combined grouped localizations for render export | "
                "channel=%d rows_before=%d rows_after=%d",
                int(ch),
                int(before_combine),
                int(len(df_export)),
            )

        if render_ready and apply_cylindrical_lens_xy_correction_for_render:
            if cylindrical_lens_xy_homography is None:
                raise ValueError(
                    "apply_cylindrical_lens_xy_correction_for_render=True, but no "
                    "cylindrical_lens_xy_homography path was provided."
                )

            if pixelsize_nm is None:
                raise ValueError(
                    "apply_cylindrical_lens_xy_correction_for_render=True, but "
                    "pixelsize_nm was not provided."
                )

            df_export = apply_xy_homography_to_localization_dataframe(
                df_export,
                cylindrical_lens_xy_homography,
                pixelsize_nm=float(pixelsize_nm),
            )

        if render_ready:
            df_out = to_render_ready_dataframe(
                df_export,
                channel_col=channel_col,
                frame_offset=int(render_frame_offset),
                z_sign=float(render_z_sign),
                z_scale=float(render_z_scale),
                uncertainty_xy_scale=float(render_uncertainty_xy_scale),
                uncertainty_z_scale=float(render_uncertainty_z_scale),
            )
        else:
            df_out = df_export

        df_out.to_csv(out_csv, index=False)
        written[int(ch)] = out_csv

        logger.info(
            "Saved channel %d localization CSV: %s | rows=%d",
            int(ch),
            out_csv,
            int(len(df_out)),
        )

    return written

def run_rcc_drift_correction(
    csv_file: str | Path,
    *,
    output_file: str | Path | None = None,
    drift_table_file: str | Path | None = None,
    frame_col: str = "frame",
    x_col: str = "x [nm]",
    y_col: str = "y [nm]",
    z_col: str = "z [nm]",
    correct_xy: bool = True,
    correct_z: bool = True,
    overwrite: bool = True,

    # Tunable RCC settings exposed from main Settings
    xy_timepoints: int = 20,
    xy_pixel_size_nm: float = 5.0,
    z_timepoints: int = 20,
    z_bin_width_nm: float = 5.0,
    max_drift_nm: float = 1000.0,
    z_range_nm: tuple[float, float] = (-400.0, 400.0),

    # Channel split export for ThunderSTORM/ImageJ workflows
    save_channel_split_csvs: bool = True,
    channel_split_channels: tuple[int, ...] = (0, 1, 2, 3, 4),
    channel_split_output_dir: str | Path | None = None,
    channel_split_output_stem: str | None = None,
    render_ready_channel_csvs: bool = True,
    render_max_number_in_group: int | None = 5,
    combine_grouped_localizations_for_render: bool = True,
    render_frame_offset: int = 1,
    render_z_sign: float = 1.0,
    render_z_scale: float = 1.0,
    render_uncertainty_xy_scale: float = 1.0,
    render_uncertainty_z_scale: float = 1.0,
    apply_cylindrical_lens_xy_correction_for_render: bool = False,
    cylindrical_lens_xy_homography: str | Path | None = None,
    pixelsize_nm: float | None = None,

    # Batch-safety behavior
    skip_on_failure: bool = True,
) -> Path:
    """
    Run RCC drift correction on a localization CSV.

    The input CSV is expected to already be filtered and, optionally, to contain
    grouping metadata such as groupindex / numberInGroup. These extra columns are
    preserved unchanged.

    If skip_on_failure=True, RCC estimation failures do not crash the pipeline.
    The original uncorrected CSV path is returned instead.

    Returns
    -------
    Path
        Path to the drift-corrected CSV, or the original CSV if RCC was skipped.
    """
    csv_file = Path(csv_file)

    if output_file is None:
        output_file = csv_file.with_name(csv_file.stem + "_rcc_drift_corrected.csv")
    else:
        output_file = Path(output_file)

    if drift_table_file is None:
        drift_table_file = csv_file.with_name(csv_file.stem + "_rcc_drift.csv")
    else:
        drift_table_file = Path(drift_table_file)

    if output_file.exists() and not overwrite:
        logger.info("RCC drift-corrected file already exists, skipping: %s", output_file)
        return output_file

    if not csv_file.exists():
        raise FileNotFoundError(f"Localization CSV not found:\n{csv_file}")

    logger.info("Loading localizations for RCC drift correction: %s", csv_file)
    locs = pd.read_csv(csv_file)

    required_cols = [frame_col, x_col, y_col]
    if correct_z:
        required_cols.append(z_col)

    missing = [col for col in required_cols if col not in locs.columns]
    if missing:
        raise ValueError(
            "Cannot run RCC drift correction. Missing column(s): "
            + ", ".join(missing)
            + f"\n\nAvailable columns:\n{list(locs.columns)}"
        )

    if locs.empty:
        msg = f"Cannot run RCC drift correction: localization CSV is empty: {csv_file}"
        if skip_on_failure:
            logger.warning("%s. Continuing with uncorrected CSV.", msg)
            return csv_file
        raise ValueError(msg)

    locs[frame_col] = locs[frame_col].astype(int)

    z0, z1 = z_range_nm

    config = RCCDriftConfig(
        correct_xy=bool(correct_xy),
        correct_z=bool(correct_z),

        xy_timepoints=int(xy_timepoints),
        xy_pixel_size_nm=float(xy_pixel_size_nm),
        xy_peak_window_pix=7,

        z_timepoints=int(z_timepoints),
        z_bin_width_nm=float(z_bin_width_nm),
        z_peak_window_pix=9,
        z_range_nm=(float(z0), float(z1)),
        z_slice_width_nm=200.0,

        max_drift_nm=float(max_drift_nm),
        max_reconstruction_size_pix=4096,

        smooth_mode="spline",
        require_min_locs=True,
    )

    logger.info(
        "Running RCC drift correction | "
        "xy=%s z=%s xy_timepoints=%d xy_pixel=%.3f nm "
        "z_timepoints=%d z_bin=%.3f nm max_drift=%.1f nm z_range=(%.1f, %.1f)",
        bool(correct_xy),
        bool(correct_z),
        int(xy_timepoints),
        float(xy_pixel_size_nm),
        int(z_timepoints),
        float(z_bin_width_nm),
        float(max_drift_nm),
        float(z0),
        float(z1),
    )

    try:
        corrected, drift_df, _ = correct_drift_xyz(
            locs,
            config=config,
            frame_col=frame_col,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
        )
    except Exception as e:
        if skip_on_failure:
            logger.warning(
                "RCC drift correction skipped for %s because drift could not be estimated. "
                "Continuing with uncorrected CSV. Reason: %s",
                csv_file,
                e,
            )
            return csv_file
        raise

    corrected.to_csv(output_file, index=False)
    logger.info("Saved RCC drift-corrected localizations: %s", output_file)

    drift_df.to_csv(drift_table_file, index=False)
    logger.info("Saved RCC drift table: %s", drift_table_file)

    if save_channel_split_csvs:
        channel_csvs = write_channel_split_csvs(
            corrected,
            output_file,
            channel_col="channel",
            channels=tuple(channel_split_channels),
            output_dir=channel_split_output_dir,
            output_stem=channel_split_output_stem,
            overwrite=overwrite,
            render_ready=bool(render_ready_channel_csvs),
            render_max_number_in_group=render_max_number_in_group,
            combine_grouped_localizations_for_render=bool(
                combine_grouped_localizations_for_render
            ),
            render_frame_offset=int(render_frame_offset),
            render_z_sign=float(render_z_sign),
            render_z_scale=float(render_z_scale),
            render_uncertainty_xy_scale=float(render_uncertainty_xy_scale),
            render_uncertainty_z_scale=float(render_uncertainty_z_scale),
            apply_cylindrical_lens_xy_correction_for_render=bool(
                apply_cylindrical_lens_xy_correction_for_render
            ),
            cylindrical_lens_xy_homography=cylindrical_lens_xy_homography,
            pixelsize_nm=pixelsize_nm,
        )
        if channel_csvs:
            logger.info(
                "Saved RCC channel-split localization CSVs: %s",
                ", ".join(str(p) for p in channel_csvs.values()),
            )

    return output_file

def run_comet_drift_correction(
    csv_file: str | Path,
    *,
    output_file: str | Path | None = None,
    drift_table_file: str | Path | None = None,
    frame_col: str = "frame",
    x_col: str = "x [nm]",
    y_col: str = "y [nm]",
    z_col: str = "z [nm]",
    overwrite: bool = True,

    # COMET settings
    segmentation_mode: int = 2,
    segmentation_var: int = 60,
    initial_sigma_nm: float = 100.0,
    target_sigma_nm: float = 1.0,
    max_drift_nm: float = 300.0,
    boxcar_width: int = 1,
    interpolation_method: str = "cubic",
    max_locs_per_segment: int | None = None,
    suppress_comet_output: bool = True,

    # Channel split export for ThunderSTORM/ImageJ workflows
    save_channel_split_csvs: bool = True,
    channel_split_channels: tuple[int, ...] = (0, 1, 2, 3, 4),
    channel_split_output_dir: str | Path | None = None,
    channel_split_output_stem: str | None = None,
    render_ready_channel_csvs: bool = True,
    render_max_number_in_group: int | None = 5,
    combine_grouped_localizations_for_render: bool = True,
    render_frame_offset: int = 1,
    render_z_sign: float = 1.0,
    render_z_scale: float = 1.0,
    render_uncertainty_xy_scale: float = 1.0,
    render_uncertainty_z_scale: float = 1.0,
    apply_cylindrical_lens_xy_correction_for_render: bool = False,
    cylindrical_lens_xy_homography: str | Path | None = None,
    pixelsize_nm: float | None = None,

    # Batch-safety behavior
    skip_on_failure: bool = True,
) -> Path:
    """
    Run COMET drift correction on a localization CSV.

    If this function is called after RCC, COMET estimates residual drift on the
    RCC-corrected localizations. Extra localization columns are preserved by
    copying corrected x/y/z values back into the original DataFrame instead of
    using COMET's CSV writer.
    """
    csv_file = Path(csv_file)

    if output_file is None:
        output_file = csv_file.with_name(csv_file.stem + "_comet_drift_corrected.csv")
    else:
        output_file = Path(output_file)

    if drift_table_file is None:
        drift_table_file = csv_file.with_name(csv_file.stem + "_comet_residual_drift.csv")
    else:
        drift_table_file = Path(drift_table_file)

    if output_file.exists() and not overwrite:
        logger.info("COMET drift-corrected file already exists, skipping: %s", output_file)
        return output_file

    if not csv_file.exists():
        raise FileNotFoundError(f"Localization CSV not found:\n{csv_file}")

    logger.info("Loading localizations for COMET drift correction: %s", csv_file)
    locs = pd.read_csv(csv_file)

    required_cols = [frame_col, x_col, y_col]
    missing = [col for col in required_cols if col not in locs.columns]
    if missing:
        raise ValueError(
            "Cannot run COMET drift correction. Missing column(s): "
            + ", ".join(missing)
            + f"\n\nAvailable columns:\n{list(locs.columns)}"
        )

    if locs.empty:
        msg = f"Cannot run COMET drift correction: localization CSV is empty: {csv_file}"
        if skip_on_failure:
            logger.warning("%s. Continuing with input CSV.", msg)
            return csv_file
        raise ValueError(msg)

    try:
        from comet.core.drift_optimizer import comet_run_kd
    except Exception as e:
        msg = (
            "Cannot import COMET. Install it from the COMET Python_interface folder, "
            "for example: pip install -e ."
        )
        if skip_on_failure:
            logger.warning("%s Continuing with input CSV. Reason: %s", msg, e)
            return csv_file
        raise ImportError(msg) from e

    try:
        frames = pd.to_numeric(locs[frame_col], errors="raise").astype(int).to_numpy()
        x_nm = pd.to_numeric(locs[x_col], errors="raise").astype(float).to_numpy()
        y_nm = pd.to_numeric(locs[y_col], errors="raise").astype(float).to_numpy()

        if z_col in locs.columns:
            z_nm = pd.to_numeric(locs[z_col], errors="coerce").astype(float).to_numpy()
            z_nm = np.where(np.isfinite(z_nm), z_nm, 0.0)
            has_z_col = True
        else:
            z_nm = np.zeros(len(locs), dtype=float)
            has_z_col = False

        dataset = np.column_stack([x_nm, y_nm, z_nm, frames]).astype(np.float64, copy=False)

        if not np.isfinite(dataset).all():
            raise ValueError("COMET input contains non-finite frame/x/y/z values.")

        logger.info(
            "Running COMET drift correction | segmentation_mode=%d segmentation_var=%d "
            "initial_sigma=%.3f nm target_sigma=%.3f nm max_drift=%.3f nm "
            "boxcar_width=%d interpolation=%s max_locs_per_segment=%s",
            int(segmentation_mode),
            int(segmentation_var),
            float(initial_sigma_nm),
            float(target_sigma_nm),
            float(max_drift_nm),
            int(boxcar_width),
            str(interpolation_method),
            str(max_locs_per_segment),
        )

        comet_kwargs = dict(
            segmentation_mode=int(segmentation_mode),
            segmentation_var=int(segmentation_var),
            max_locs_per_segment=max_locs_per_segment,
            initial_sigma_nm=float(initial_sigma_nm),
            target_sigma_nm=float(target_sigma_nm),
            max_drift_nm=float(max_drift_nm),
            boxcar_width=int(boxcar_width),
            interpolation_method=str(interpolation_method),
            return_corrected_locs=True,
        )

        if suppress_comet_output:
            captured_output = io.StringIO()
            with (
                contextlib.redirect_stdout(captured_output),
                contextlib.redirect_stderr(captured_output),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                drift, corrected_arr = comet_run_kd(dataset.copy(), **comet_kwargs)

            captured_text = captured_output.getvalue()
            if captured_text:
                logger.debug("Suppressed COMET output:\n%s", captured_text)
        else:
            drift, corrected_arr = comet_run_kd(dataset.copy(), **comet_kwargs)

        corrected_arr = np.asarray(corrected_arr)
        if corrected_arr.shape[0] != len(locs) or corrected_arr.shape[1] < 4:
            raise RuntimeError(
                "COMET returned corrected localizations with an unexpected shape: "
                f"{corrected_arr.shape}"
            )

        corrected = locs.copy()
        corrected[x_col] = corrected_arr[:, 0]
        corrected[y_col] = corrected_arr[:, 1]
        if has_z_col:
            corrected[z_col] = corrected_arr[:, 2]

        drift = np.asarray(drift)
        if drift.ndim != 2 or drift.shape[1] < 4:
            raise RuntimeError(f"COMET returned drift with an unexpected shape: {drift.shape}")

        drift_df = pd.DataFrame(
            {
                "frame": drift[:, 3].astype(int),
                "dx_nm": drift[:, 0],
                "dy_nm": drift[:, 1],
                "dz_nm": drift[:, 2],
            }
        )

    except Exception as e:
        if skip_on_failure:
            logger.warning(
                "COMET drift correction skipped for %s because drift could not be estimated. "
                "Continuing with input CSV. Reason: %s",
                csv_file,
                e,
            )
            return csv_file
        raise

    corrected.to_csv(output_file, index=False)
    logger.info("Saved COMET drift-corrected localizations: %s", output_file)

    drift_df.to_csv(drift_table_file, index=False)
    logger.info("Saved COMET residual drift table: %s", drift_table_file)

    if save_channel_split_csvs:
        channel_csvs = write_channel_split_csvs(
            corrected,
            output_file,
            channel_col="channel",
            channels=tuple(channel_split_channels),
            output_dir=channel_split_output_dir,
            output_stem=channel_split_output_stem,
            overwrite=overwrite,
            render_ready=bool(render_ready_channel_csvs),
            render_max_number_in_group=render_max_number_in_group,
            combine_grouped_localizations_for_render=bool(
                combine_grouped_localizations_for_render
            ),
            render_frame_offset=int(render_frame_offset),
            render_z_sign=float(render_z_sign),
            render_z_scale=float(render_z_scale),
            render_uncertainty_xy_scale=float(render_uncertainty_xy_scale),
            render_uncertainty_z_scale=float(render_uncertainty_z_scale),
            apply_cylindrical_lens_xy_correction_for_render=bool(
                apply_cylindrical_lens_xy_correction_for_render
            ),
            cylindrical_lens_xy_homography=cylindrical_lens_xy_homography,
            pixelsize_nm=pixelsize_nm,
        )
        if channel_csvs:
            logger.info(
                "Saved COMET channel-split localization CSVs: %s",
                ", ".join(str(p) for p in channel_csvs.values()),
            )

    return output_file