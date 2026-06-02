# mle_fitting_drivers.py

"""
GPU fitting pipeline
--------------------
High-level GPU fitting orchestration for the ratiometric STORM pipeline.

Function groups
---------------
Stage 1 model preparation
    Load and pack per-channel spline PSF models for loclib.

Stage-level drivers
    Run complete pipeline stages over TIFF/CSV inputs and write CSV outputs.

Stage 3 GPU-batch model fitters
    Fit one already-prepared batch of paired R/T ROIs.

Stage 3 result formatting
    Call helper formatters that convert raw GPU arrays into export-ready
    localization columns.

Key array contracts
-------------------
- Preprocessing cuts image-order ROIs as (N, Y, X), matching NumPy image
  indexing image[y, x].
- Stage-level fitting drivers pass image-order ROI batches into GPU helper
  functions. The helper layer performs loclib/CUDA memory packing immediately
  before calling the DLL.
- Single-channel spline coefficients are packed as (B, Z, Y, X), with matching
  spline-size metadata [X, Y, Z, B].
- Global dual-channel fitting receives paired R/T image-order ROIs generated from
  Stage 1 CSVs after T→R transformation, R/T pairing, inverse-variance coordinate
  merging, residual preservation, and paired ROI cutout.
- The multichannel loclib wrapper packs paired ROIs as (C, N, X, Y), matching the
  CUDA convention data(x, y, fit, channel).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from .stage3_pipeline_helpers import (
    append_stage3_full_csv,
    CsvBatchWriter,
    finalize_stage3_filtered_csv,
    make_stage3_paired_roi_batch_iterator,
    prepare_stage3_context,
    prepare_stage3_temp_csv_paths,
    rename_stage3_csv_if_written,
    unpack_stage3_paired_roi_batch,
)

from .mle_fitting_helpers import (
    build_dts_dual_channel_free_ratio,
    build_dts_dual_channel_fixed_ratio,
    coeff_xyzb_matlab_to_single_channel_loclib_bzyx,
    build_single_channel_splinesize_for_dll,
    mle_spline_single_channel,
    mle_spline_dual_channel,
    format_single_channel_fit_results,
    format_global_fit_results_free_ratio,
    format_global_fit_results_fixed_ratio,
)

from .pipeline_config import (
    fixed_ratios_as_t_over_r,
    psf_model_format_normalized,
)

from .preprocessing_utils import load_channel_psf_models

from .uipsf_model_loader import load_uipsf_coeff_tensor

from .roi_preparation import (
    iter_single_channel_roi_batches_from_tiff,
)

logger = logging.getLogger("stormpipe.gpu.pipeline")
SPLINE_PARAM_COUNT = 6  # [Y, X, Phot, BG, Zidx, Iter]


# =============================================================================
# Stage 1 model preparation
# =============================================================================
@dataclass(frozen=True)
class ChannelModel:
    # Loclib-prepared single-channel spline coefficients:
    #     coeff.shape == (B, Z, Y, X)
    #
    # This is the same per-channel coefficient layout used by the global
    # dual-channel model before channel stacking.
    coeff: np.ndarray

    # Single-channel spline-size metadata:
    #     splinesize == [X, Y, Z, B]
    #
    # The coefficient tensor and splinesize metadata must be passed to the DLL
    # together as a matched pair.
    splinesize: np.ndarray
    zseed: np.float32
    dz: float
    z0: int
    mirror: bool
    normf: float


def prepare_channel_models(settings: "Settings") -> dict[str, ChannelModel]:
    """
    Load per-channel Stage 1 spline PSF models and prepare them for loclib.

    SMAP mode:
        Load MATLAB/SMAP per-channel coeffs in logical layout (X,Y,Z,64),
        then pack each channel to (64,Z,Y,X).

    uiPSF mode:
        Load /locres/<uipsf_coeff_key> from the uiPSF .h5 model. This is expected
        to already be packed as (2,64,Z,Y,X). Slice channel 0/1 directly into
        per-channel Stage 1 models.
    """
    fmt = psf_model_format_normalized(settings)

    if fmt == "uiPSF":
        return _prepare_uipsf_channel_models(settings)

    if fmt != "SMAP":
        raise ValueError(f"Unsupported psf_model_format={fmt!r}")

    psf = load_channel_psf_models(settings.psf_model, settings.channel_map, verbose=False)
    out: dict[str, ChannelModel] = {}

    for tag in ("R", "T"):
        psf[tag]["mirror"] = False

        coeff_loclib = coeff_xyzb_matlab_to_single_channel_loclib_bzyx(
            psf[tag]["coeff"]
        )

        spl = build_single_channel_splinesize_for_dll(coeff_loclib)
        zseed = np.float32(psf[tag]["z0"] + 1e-6)

        logger.info(
            "%s coeff layout: SMAP single_channel_bzyx | coeff shape %s | splinesize=%s",
            tag,
            coeff_loclib.shape,
            spl.tolist(),
        )

        out[tag] = ChannelModel(
            coeff=coeff_loclib,
            splinesize=np.asarray(spl, dtype=np.int32),
            zseed=zseed,
            dz=float(psf[tag]["dz"]),
            z0=int(psf[tag]["z0"]),
            mirror=bool(psf[tag]["mirror"]),
            normf=float(psf[tag]["normf"]),
        )

    return out

def _prepare_uipsf_channel_models(settings: "Settings") -> dict[str, ChannelModel]:
    """
    Prepare Stage 1 per-channel models from a uiPSF dual-channel .h5 model.

    Expected uiPSF coefficient layout:
        locres/<uipsf_coeff_key>.shape == (2, 64, Z, Y, X)

    Channel mapping follows settings.channel_map:
        R -> 0
        T -> 1
    by default.
    """
    data = load_uipsf_coeff_tensor(
        settings.psf_model,
        coeff_key=settings.uipsf_coeff_key,
        z0_index=settings.uipsf_z0_index,
        normf=settings.uipsf_normf,
        swap_xy_axes=settings.uipsf_swap_xy_axes,
        verbose=True,
    )

    out: dict[str, ChannelModel] = {}

    for tag in ("R", "T"):
        ch_idx = int(settings.channel_map[tag])
        if ch_idx not in (0, 1):
            raise ValueError(
                f"uiPSF Stage 1 supports channel indices 0 and 1 only; "
                f"settings.channel_map[{tag!r}]={ch_idx}"
            )

        coeff_loclib = np.ascontiguousarray(data.coeff[ch_idx], dtype=np.float32)
        spl = build_single_channel_splinesize_for_dll(coeff_loclib)

        logger.info(
            "%s coeff layout: uiPSF single_channel_bzyx | source channel=%d | "
            "coeff shape %s | splinesize=%s | dz=%.6g nm | z0=%d | normf=%.6g",
            tag,
            ch_idx,
            coeff_loclib.shape,
            spl.tolist(),
            float(data.dz),
            int(data.z0),
            float(data.normf[ch_idx]),
        )

        out[tag] = ChannelModel(
            coeff=coeff_loclib,
            splinesize=np.asarray(spl, dtype=np.int32),
            zseed=np.float32(data.zseed),
            dz=float(data.dz),
            z0=int(data.z0),
            mirror=bool(data.mirror),
            normf=float(data.normf[ch_idx]),
        )

    return out

# =============================================================================
# Stage-level drivers
# =============================================================================
def fit_stage1_single_channel(
    tag: str,
    tif_path: str | Path,
    model: ChannelModel,
    settings: "Settings",
    dll,
    flush_every: int = 10,
    *,
    out_csv: str | Path | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> Path:
    """
    Run the Stage 1 single-channel fitting driver for one TIFF channel.

    This function iterates through one channel TIFF frame-by-frame: detecting
    candidate peaks using that channel's threshold, cuts ROIs, calls the
    single-channel spline MLE GPU fitter, formats the results, and writes <tag>_fits.csv.

    The main pipeline calls this once for reflected ("R") and once for transmitted
    ("T"), over the full TIFF stack.

    Parameters
    ----------
    tag
        Channel tag, usually "R" or "T".
    tif_path
        TIFF stack for this channel.
    model
        Prepared single-channel spline PSF model for this channel.
    settings
        Pipeline settings object.
    dll
        loclib localizationlib instance.
    start_frame, max_frames
        Optional frame-window controls for partial-stack fitting.
    """
    if out_csv is None:
        out_csv = settings.output_dir() / f"{tag}_fits.csv"
    else:
        out_csv = Path(out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = CsvBatchWriter(out_csv, flush_every=flush_every)

    Nmax = int(settings.gpu_batch_size)
    K = SPLINE_PARAM_COUNT  # [Y, X, Phot, BG, Zidx, Iter]
    P_buf    = np.zeros((K,   Nmax), dtype=np.float32, order="F")
    CRLB_buf = np.zeros((K-1, Nmax), dtype=np.float32, order="F")
    LL_buf   = np.zeros((Nmax,),     dtype=np.float32)

    total = 0

    import tifffile

    with tifffile.TiffFile(str(tif_path)) as tif:
        n_total_frames = len(tif.pages)

    f0 = int(max(0, start_frame))
    fend = n_total_frames if max_frames is None else min(n_total_frames, f0 + int(max_frames))
    total_fit_frames = max(0, fend - f0)

    end_frame_str = str(fend)
    channel_name = "reflected" if tag == "R" else "transmitted"

    logger.info(
        "%s Stage 1 fit starting: detecting candidate ROIs in the %s channel "
        "and fitting them with the %s single-channel spline PSF model | frames [%s, %s)",
        tag,
        channel_name,
        channel_name,
        start_frame,
        end_frame_str,
    )

    thr_this = settings.thr_factor_for_channel(tag)  # independent detection threshold per channel

    with tqdm(
        total=total_fit_frames,
        desc=f"Stage 1 {tag} frames",
        unit="frame",
        position=0,
        leave=True,
    ) as frame_pbar:

        for rois, xpix, ypix, frames, _ids in iter_single_channel_roi_batches_from_tiff(
            tif_path,
            offset=settings.offset,
            camera_gain=settings.camera_gain,
            QE=settings.qe,
            roi_size=settings.roi_size,
            thr=thr_this,
            s1=settings.sigma1,
            s2=settings.sigma2,
            keep_peak_winner=settings.keep_peak_winner,
            mindist=settings.min_distance,
            batch_size=settings.gpu_batch_size,
            max_frames=max_frames,
            start_frame=start_frame,
            roi_id_start=0,
            on_frame=lambda _f: frame_pbar.update(1),
        ):
            if rois.shape[0] == 0:
                continue

            assert rois.shape[1:] == (settings.roi_size, settings.roi_size), \
                f"ROI shape mismatch: {rois.shape}, expected {(settings.roi_size, settings.roi_size)}"
            total += rois.shape[0]

            P, CRLB, LL, _ = mle_spline_single_channel(
                dll,
                rois,
                xpix,
                ypix,
                frames,
                coeff_loclib_single=model.coeff,
                splinesize_loclib_single=model.splinesize,
                iterations=int(settings.gpu_iterations),
                EMexcessNoise=int(settings.em_excess_noise),
                varstack=0,
                zseed_abs=model.zseed,
                prealloc_P=P_buf[:, :rois.shape[0]],
                prealloc_CRLB=CRLB_buf[:, :rois.shape[0]],
                prealloc_LogL=LL_buf[:rois.shape[0]],
            )

            cols = format_single_channel_fit_results(
                P, CRLB, LL, xpix, ypix, frames,
                roi_size=settings.roi_size,
                dz_nm=model.dz, z0_index=model.z0,
                mirror=model.mirror, normf=model.normf,
                EMexcessNoise=settings.em_excess_noise,
                RI_mismatch=settings.ri_mismatch,
                pixelsize_nm=settings.pixelsize_nm,
            )

            result_df = pd.DataFrame(cols)
            writer.write(result_df)

    writer.finalize()

    logger.info(
        "%s Stage 1 fit complete: fitted %d candidate ROIs in the %s channel "
        "with the %s single-channel spline PSF model → %s | frames [%s, %s)",
        tag,
        total,
        channel_name,
        channel_name,
        out_csv,
        start_frame,
        end_frame_str,
    )
    return out_csv


def fit_global_stage3_free_ratio(
    tif_R_path: str | Path,
    tif_T_path: str | Path,
    H_T_to_R: np.ndarray | str | Path,
    settings,
    dll,
    *,
    model=None,
    out_csv: str | Path | None = None,
    outdir: str | Path | None = None,
    R_stage1_csv: str | Path | None = None,
    T_stage1_csv: str | Path | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    pair_stem: str | None = None,
) -> Path:
    """
    Run the Stage 3 global fitting driver in free-ratio mode.
    R and T photon counts are fitted independently.

    Stage 3 paired ROIs are generated from Stage 1 CSV localizations. The CPU-side
    paired-ROI iterator transforms T localizations into R space, pairs R/T localizations,
    merges paired coordinates by inverse-variance weighting, preserves per-channel
    subpixel residuals, cuts paired R/T ROIs, and yields those batches to this GPU
    driver. This driver then calls the free-ratio GPU fitter for each batch, formats
    the results, writes the full CSV, applies standard Stage 3 filtering/grouping,
    and writes the filtered/grouped CSV.
    """

    from .stage3_filtering_grouping import (
        filter_stage3_dataframe,
    )
    ctx = prepare_stage3_context(
        tif_R_path,
        tif_T_path,
        H_T_to_R,
        settings,
        model=model,
        outdir=outdir,
        R_stage1_csv=R_stage1_csv,
        T_stage1_csv=T_stage1_csv,
        start_frame=start_frame,
        max_frames=max_frames,
        pair_stem=pair_stem,
    )

    shared_stem = ctx.shared_stem
    outdir = ctx.outdir
    model = ctx.model
    total_stage3_frames = ctx.total_stage3_frames

    # Temporary write paths; final names are assigned after row counts are known
    csv_paths = prepare_stage3_temp_csv_paths(
        outdir,
        mode_name="free_ratio",
        out_csv=out_csv,
    )

    full_csv_tmp = csv_paths.full_csv_tmp
    filtered_csv_tmp = csv_paths.filtered_csv_tmp

    full_csv = full_csv_tmp
    filtered_csv = filtered_csv_tmp

    full_header_written = False
    total_pairs_seen = 0
    total_rows_written_full = 0
    batch_index = 0

    filtered_result_dfs: list[pd.DataFrame] = []

    with tqdm(
        total=total_stage3_frames,
        desc="Stage 3 frames",
        unit="frame",
        position=0,
    ) as frame_pbar, tqdm(
        desc="Stage 3 paired ROIs",
        unit="pair",
        position=1,
        leave=True,
    ) as pair_pbar:

        roi_batch_iter = make_stage3_paired_roi_batch_iterator(
            ctx,
            settings,
            start_frame=start_frame,
            max_frames=max_frames,
            on_frame=lambda _f: frame_pbar.update(1),
        )

        for batch_tuple in roi_batch_iter:
            batch = unpack_stage3_paired_roi_batch(batch_tuple)

            if batch.n_pairs == 0:
                continue

            batch_index += 1
            N = batch.n_pairs
            total_pairs_seen += N

            P, CRLB, LL, _nfit = fit_global_stage3_gpu_batch_free_ratio(
                dll,
                rois_R=batch.rois_R,
                rois_T=batch.rois_T,
                R_dxdy=batch.R_dxdy,
                T_dxdy=batch.T_dxdy,
                model=model,
                iterations=settings.gpu_iterations,
                zseed_abs=batch.zseed_abs,
            )

            cols = format_global_fit_results_free_ratio(
                P,
                CRLB,
                LL,
                R_centers=batch.R_centers,
                T_centers=batch.T_centers,
                R_dxdy=batch.R_dxdy,
                T_dxdy=batch.T_dxdy,
                frames=batch.frames,
                roi_size=settings.roi_size,
                dz_nm=model.dz,
                z0_index=model.z0,
                normf=model.normf,
                EMexcessNoise=settings.em_excess_noise,
                RI_mismatch=settings.ri_mismatch,
                pixelsize_nm=settings.pixelsize_nm,
                main_channel=settings.main_channel,
                free_ratio_output_channel=settings.free_ratio_output_channel,
            )

            result_df = pd.DataFrame(cols)
            result_df.insert(0, "pair_id", batch.pair_ids.astype(np.int64, copy=False))

            full_header_written, total_rows_written_full = append_stage3_full_csv(
                result_df,
                full_csv_tmp,
                settings,
                header_written=full_header_written,
                rows_written=total_rows_written_full,
            )

            if getattr(settings, "save_stage3_filtered_csv", True) and not result_df.empty:
                filtered_result_df = filter_stage3_dataframe(result_df, settings)
                if not filtered_result_df.empty:
                    filtered_result_dfs.append(filtered_result_df)

            pair_pbar.update(N)

    total_rows_written_filtered, _df_all = finalize_stage3_filtered_csv(
        filtered_result_dfs,
        filtered_csv_tmp,
        settings,
    )

    logger.info("Finalizing output filenames.")

    final_full_name = f"{shared_stem}_locs_free_full.csv"
    full_csv = rename_stage3_csv_if_written(
        full_csv_tmp,
        outdir / final_full_name,
        rows_written=total_rows_written_full,
        enabled=getattr(settings, "save_stage3_full_csv", True),
        label="full CSV",
    )

    final_filtered_name = f"{shared_stem}_locs_free_filtered.csv"
    filtered_csv = rename_stage3_csv_if_written(
        filtered_csv_tmp,
        outdir / final_filtered_name,
        rows_written=total_rows_written_filtered,
        enabled=getattr(settings, "save_stage3_filtered_csv", True),
        label="filtered CSV",
    )

    logger.info(
        "Stage 3 free-ratio global fit done: "
        "batches=%d pairs_seen=%d full_rows=%d filtered_rows=%d full_csv=%s filtered_csv=%s",
        batch_index,
        total_pairs_seen,
        total_rows_written_full,
        total_rows_written_filtered,
        full_csv,
        filtered_csv,
    )

    if getattr(settings, "save_stage3_filtered_csv", True) and total_rows_written_filtered > 0:
        return filtered_csv
    return full_csv


def fit_global_stage3_fixed_ratios(
    tif_R_path: str | Path,
    tif_T_path: str | Path,
    H_T_to_R: np.ndarray | str | Path,
    settings,
    dll,
    *,
    model=None,
    out_csv: str | Path | None = None,
    outdir: str | Path | None = None,
    R_stage1_csv: str | Path | None = None,
    T_stage1_csv: str | Path | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    pair_stem: str | None = None,
) -> Path:
    """
    Run the Stage 3 global fitting driver in fixed-ratio mode.
    The T/R photon ratio is constrained to one of settings.fixed_ratios.
    (Provided in settings as T/(R+T) for user convenience, but converted internally to T/R
    for fitting, as required by the MLE globloc compiled cuda code.)
    
    Stage 3 paired ROIs are generated from Stage 1 CSV localizations. The CPU-side
    paired-ROI iterator transforms T localizations into R space, pairs R/T localizations,
    merges paired coordinates by inverse-variance weighting, preserves per-channel
    subpixel residuals, cuts paired R/T ROIs, and yields those batches to this GPU
    driver.
    For each paired ROI batch, this function runs one constrained global fit per
    candidate fixed ratio, selects the best ratio by log-likelihood, formats results,
    writes the full CSV, applies standard Stage 3 filtering/grouping, optionally
    rejects ambiguous ratio assignments, and writes the filtered/grouped CSV.
    """

    from .stage3_filtering_grouping import (
        filter_stage3_dataframe,
        filter_fixed_ratio_assignment_ambiguity,
    )
    ctx = prepare_stage3_context(
        tif_R_path,
        tif_T_path,
        H_T_to_R,
        settings,
        model=model,
        outdir=outdir,
        R_stage1_csv=R_stage1_csv,
        T_stage1_csv=T_stage1_csv,
        start_frame=start_frame,
        max_frames=max_frames,
        pair_stem=pair_stem,
    )
    shared_stem = ctx.shared_stem
    outdir = ctx.outdir
    model = ctx.model
    total_stage3_frames = ctx.total_stage3_frames

    ratios = np.asarray(fixed_ratios_as_t_over_r(settings), dtype=np.float32)

    # Temporary write paths; final names are assigned after row counts are known
    csv_paths = prepare_stage3_temp_csv_paths(
        outdir,
        mode_name="fixed_ratio",
        out_csv=out_csv,
    )

    full_csv_tmp = csv_paths.full_csv_tmp
    filtered_csv_tmp = csv_paths.filtered_csv_tmp

    full_csv = full_csv_tmp
    filtered_csv = filtered_csv_tmp

    full_header_written = False
    total_pairs_seen = 0
    total_rows_written_full = 0
    batch_index = 0

    filtered_result_dfs: list[pd.DataFrame] = []

    with tqdm(
        total=total_stage3_frames,
        desc="Stage 3 frames",
        unit="frame",
        position=0,
    ) as frame_pbar, tqdm(
        desc="Stage 3 paired ROIs",
        unit="pair",
        position=1,
        leave=True,
    ) as pair_pbar:

        roi_batch_iter = make_stage3_paired_roi_batch_iterator(
            ctx,
            settings,
            start_frame=start_frame,
            max_frames=max_frames,
            on_frame=lambda _f: frame_pbar.update(1),
        )

        for batch_tuple in roi_batch_iter:
            batch = unpack_stage3_paired_roi_batch(batch_tuple)

            if batch.n_pairs == 0:
                continue

            batch_index += 1
            N = batch.n_pairs
            total_pairs_seen += N

            P_list = []
            CRLB_list = []
            LL_list = []

            for ratio in ratios:
                Pk, CRLBk, LLk, _ = fit_global_stage3_gpu_batch_fixed_ratio(
                    dll,
                    rois_R=batch.rois_R,
                    rois_T=batch.rois_T,
                    R_dxdy=batch.R_dxdy,
                    T_dxdy=batch.T_dxdy,
                    model=model,
                    photon_ratio=float(ratio),
                    iterations=settings.gpu_iterations,
                    zseed_abs=batch.zseed_abs,
                )
                P_list.append(Pk)
                CRLB_list.append(CRLBk)
                LL_list.append(LLk)

            # Stack candidate-ratio fits and choose the best fixed-ratio assignment
            # independently for each paired ROI by maximum log-likelihood:
            P_stack = np.stack(P_list, axis=0)       # (R, 7, N)
            CRLB_stack = np.stack(CRLB_list, axis=0) # (R, 6, N)
            LL_stack = np.stack(LL_list, axis=0)     # (R, N)

            best_idx = np.argmax(LL_stack, axis=0)   # (N,)
            sel = np.arange(N, dtype=np.int64)

            P_best = P_stack[best_idx, :, sel].transpose(1, 0)         # (7, N)
            CRLB_best = CRLB_stack[best_idx, :, sel].transpose(1, 0)   # (6, N)
            LL_best = LL_stack[best_idx, sel]                           # (N,)

            if LL_stack.shape[0] > 1:
                LL_sorted = np.sort(LL_stack, axis=0)
                ll_second = LL_sorted[-2, :]
            else:
                ll_second = np.full((N,), np.nan, dtype=np.float32)

            assigned_ratio = ratios[best_idx]
            assigned_channel = best_idx.astype(np.int32) + 1

            cols = format_global_fit_results_fixed_ratio(
                P_best,
                CRLB_best,
                LL_best,
                R_centers=batch.R_centers,
                T_centers=batch.T_centers,
                R_dxdy=batch.R_dxdy,
                T_dxdy=batch.T_dxdy,
                frames=batch.frames,
                assigned_ratio=assigned_ratio,
                assigned_channel=assigned_channel,
                ll_second=ll_second,
                roi_size=settings.roi_size,
                dz_nm=model.dz,
                z0_index=model.z0,
                normf=model.normf,
                EMexcessNoise=settings.em_excess_noise,
                RI_mismatch=settings.ri_mismatch,
                pixelsize_nm=settings.pixelsize_nm,
                main_channel=settings.main_channel,
            )

            result_df = pd.DataFrame(cols)
            result_df.insert(0, "pair_id", batch.pair_ids.astype(np.int64, copy=False))

            full_header_written, total_rows_written_full = append_stage3_full_csv(
                result_df,
                full_csv_tmp,
                settings,
                header_written=full_header_written,
                rows_written=total_rows_written_full,
            )

            if getattr(settings, "save_stage3_filtered_csv", True) and not result_df.empty:
                filtered_result_df = filter_stage3_dataframe(result_df, settings)

                if (
                    getattr(settings, "fixed_ratio_reject_ambiguous", True)
                    and not filtered_result_df.empty
                ):
                    filtered_result_df = filter_fixed_ratio_assignment_ambiguity(
                        filtered_result_df,
                        ll_ratio_threshold=float(settings.fixed_ratio_ll_ratio_threshold),
                    )

                if not filtered_result_df.empty:
                    filtered_result_dfs.append(filtered_result_df)

            pair_pbar.update(N)

    total_rows_written_filtered, df_all = finalize_stage3_filtered_csv(
        filtered_result_dfs,
        filtered_csv_tmp,
        settings,
    )

    logger.info("Finalizing output filenames.")

    final_full_name = f"{shared_stem}_locs_fixed_full.csv"
    full_csv = rename_stage3_csv_if_written(
        full_csv_tmp,
        outdir / final_full_name,
        rows_written=total_rows_written_full,
        enabled=getattr(settings, "save_stage3_full_csv", True),
        label="full CSV",
    )

    final_filtered_name = f"{shared_stem}_locs_fixed_filtered.csv"
    filtered_csv = rename_stage3_csv_if_written(
        filtered_csv_tmp,
        outdir / final_filtered_name,
        rows_written=total_rows_written_filtered,
        enabled=getattr(settings, "save_stage3_filtered_csv", True),
        label="filtered CSV",
    )

    logger.info(
        "Stage 3 fixed-ratio global fit done: "
        "batches=%d pairs_seen=%d full_rows=%d filtered_rows=%d full_csv=%s filtered_csv=%s",
        batch_index,
        total_pairs_seen,
        total_rows_written_full,
        total_rows_written_filtered,
        full_csv,
        filtered_csv,
    )

    if getattr(settings, "save_stage3_filtered_csv", True) and total_rows_written_filtered > 0:
        return filtered_csv
    return full_csv


# =============================================================================
# Stage 3 GPU-batch model fitters
# =============================================================================
def fit_global_stage3_gpu_batch_free_ratio(
    dll,
    rois_R: np.ndarray,
    rois_T: np.ndarray,
    R_dxdy: np.ndarray,          # (N,2) [dx,dy]
    T_dxdy: np.ndarray,          # (N,2) [dx,dy]
    model,                       # GlobalDualChannelSplineModel
    *,
    iterations: int = 30,
    zseed_abs: np.ndarray | None = None,
    varim: int | np.ndarray = 0,
):
    """
    Fit one GPU batch of paired R/T ROIs using the free-ratio global model.
    R and T share xyz positions, but fit independent photon and background
    parameters.

    This is a GPU-batch model fitter, not a stage-level driver. It does not iterate over
    TIFF frames, write CSVs, filter localizations, or group results. Paired ROI fitting only.
    (The full free-ratio Stage 3 driver, fit_global_stage3_free_ratio, calls once per
    paired-ROI batch.)

    Free-ratio shared-parameter model:
        shared = [1, 1, 1, 0, 0]
                y  x  z  I  bg
    """

    rois_R = np.asarray(rois_R, dtype=np.float32)
    rois_T = np.asarray(rois_T, dtype=np.float32)

    if rois_R.ndim != 3:
        raise ValueError(f"rois_R must be image-order (N,Y,X), got {rois_R.shape}")
    if rois_T.shape != rois_R.shape:
        raise ValueError(
            f"rois_T must match rois_R image-order shape, "
            f"got {rois_T.shape} vs {rois_R.shape}"
        )

    N = rois_R.shape[0]

    shared = np.array([1, 1, 1, 0, 0], dtype=np.int32)

    dts = build_dts_dual_channel_free_ratio(
        R_dxdy=R_dxdy,
        T_dxdy=T_dxdy,
    )

    if zseed_abs is None:
        zseed_abs = np.full((N,), np.float32(model.zseed), dtype=np.float32)
    else:
        zseed_abs = np.asarray(zseed_abs, dtype=np.float32).reshape(-1)
        if zseed_abs.shape != (N,):
            raise ValueError(f"zseed_abs must be ({N},), got {zseed_abs.shape}")

    P, CRLB, LL, nfit = mle_spline_dual_channel(
        dll=dll,
        rois_R=rois_R,
        rois_T=rois_T,
        dts=dts,
        shared=shared,
        coeff_loclib_cbzyx=model.coeff,
        splinesize_loclib_xyzbc=model.splinesize,
        zseed_abs=zseed_abs,
        iterations=int(iterations),
        varim=varim,
    )

    return P, CRLB, LL, nfit


def fit_global_stage3_gpu_batch_fixed_ratio(
    dll,
    rois_R: np.ndarray,
    rois_T: np.ndarray,
    R_dxdy: np.ndarray,
    T_dxdy: np.ndarray,
    model,
    *,
    photon_ratio: float,
    iterations: int = 30,
    zseed_abs: np.ndarray | None = None,
    varim: int | np.ndarray = 0,
):
    
    """
    Fit one GPU batch of paired R/T ROIs for one fixed T/R photon ratio.
    R and T share xyz positions and photon amplitude. The T photon amplitude is
    linked to R through the supplied photon_ratio. In the Stage 3 driver, this
    value comes from settings.fixed_ratios (converted to T/R
    before this GPU-batch fitter is called.)

    This is a GPU-batch model fitter, not a stage-level driver. The full fixed-ratio
    Stage 3 driver (fit_global_stage3_fixed_ratios) calls this once per candidate
    ratio and then selects the best ratio by log-likelihood.

    Fixed-ratio shared-parameter model:
        shared = [1, 1, 1, 1, 0]
                y  x  z  I  bg
    """

    rois_R = np.asarray(rois_R, dtype=np.float32)
    rois_T = np.asarray(rois_T, dtype=np.float32)

    if rois_R.ndim != 3:
        raise ValueError(f"rois_R must be image-order (N,Y,X), got {rois_R.shape}")
    if rois_T.shape != rois_R.shape:
        raise ValueError(
            f"rois_T must match rois_R image-order shape, "
            f"got {rois_T.shape} vs {rois_R.shape}"
        )

    N = rois_R.shape[0]

    shared = np.array([1, 1, 1, 1, 0], dtype=np.int32)

    normf = np.asarray(model.normf, dtype=np.float32).reshape(-1)
    if normf.size < 2:
        normf = np.array([1.0, 1.0], dtype=np.float32)

    # photon_ratio is the exported, normalized T/R photon ratio.
    # The loclib dTS photon scale is applied before per-channel normf correction,
    # so convert the ratio into fitter-space units.
    ratio_for_dts = float(photon_ratio) * float(normf[0]) / float(normf[1])

    dts = build_dts_dual_channel_fixed_ratio(
        R_dxdy=R_dxdy,
        T_dxdy=T_dxdy,
        photon_ratio=ratio_for_dts,
    )

    if zseed_abs is None:
        zseed_abs = np.full((N,), np.float32(model.zseed), dtype=np.float32)
    else:
        zseed_abs = np.asarray(zseed_abs, dtype=np.float32).reshape(-1)
        if zseed_abs.shape != (N,):
            raise ValueError(f"zseed_abs must be ({N},), got {zseed_abs.shape}")

    P, CRLB, LL, nfit = mle_spline_dual_channel(
        dll=dll,
        rois_R=rois_R,
        rois_T=rois_T,
        dts=dts,
        shared=shared,
        coeff_loclib_cbzyx=model.coeff,
        splinesize_loclib_xyzbc=model.splinesize,
        zseed_abs=zseed_abs,
        iterations=int(iterations),
        varim=varim,
    )

    return P, CRLB, LL, nfit
