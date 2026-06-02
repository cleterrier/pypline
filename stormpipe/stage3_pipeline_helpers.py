# stage3_pipeline_helpers.py
"""
Pipeline helper utilities
-------------------------
Shared non-GPU orchestration helpers for the ratiometric STORM pipeline.

This module contains small pipeline-level utilities used by the stage drivers:
- Stage 3 run setup and path/context preparation
- Stage 3 paired-ROI GPU-batch containers
- CSV writing and output-finalization helpers
- Output filename tag helpers

Important naming convention
---------------------------
"GPU batch" means a group of paired ROI cutouts sent to the GPU fitter together.
It does not mean batch-processing many TIFF pairs from a folder. Folder-level
dataset batching should live in a separate, higher-level workflow later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import pandas as pd

from .preprocessing_utils import (
    count_readable_tiff_prefix,
)

from .roi_preparation import (
    iter_stage3_paired_roi_batches_from_stage1_csvs,
)
from .mle_fitting_helpers import (
    load_global_dual_channel_spline_model,
)
from .pipeline_config import psf_model_format_normalized
from .uipsf_model_loader import load_uipsf_global_dual_channel_spline_model

logger = logging.getLogger("stormpipe.pipeline_helpers")

# =============================================================================
# Stage 3 run setup and input preparation
# =============================================================================
@dataclass(frozen=True)
class Stage3RunContext:
    """
    Shared setup state for one Stage 3 global-fitting run.

    One context corresponds to one reflected/transmitted TIFF pair and one T-to-R
    homography. It stores resolved input paths, output paths, the global dual-channel
    spline PSF model, and the number of overlapping frames visited by the Stage 3
    paired-ROI iterator.

    Stage 3 reports final global coordinates in reflected-channel reference space.
    """
    tif_R_path: Path
    tif_T_path: Path
    shared_stem: str
    H_T_to_R: np.ndarray
    outdir: Path
    R_stage1_csv: Path
    T_stage1_csv: Path
    model: object
    total_stage3_frames: int


def _load_homography_input(H_T_to_R: np.ndarray | str | Path) -> np.ndarray:
    """
    Load or validate the homography that maps transmitted-channel coordinates
    into reflected-channel coordinates.

    Parameters
    ----------
    H_T_to_R
        Either an already-loaded 3x3 numpy array, a .npy path, or a text path
        readable by np.loadtxt.

    Returns
    -------
    np.ndarray
        Float64 homography matrix with shape (3, 3), mapping T pixel coordinates
        into R pixel coordinate space.
    """
    if isinstance(H_T_to_R, np.ndarray):
        H = np.asarray(H_T_to_R, dtype=np.float64)
    else:
        p = Path(H_T_to_R)
        if not p.exists():
            raise FileNotFoundError(f"Homography file not found: {p}")
        if p.suffix.lower() == ".npy":
            H = np.load(p).astype(np.float64)
        else:
            H = np.loadtxt(p).astype(np.float64)

    # Stage 3 pairing assumes this transform maps transmitted-channel pixel
    # coordinates into reflected-channel pixel coordinates.
    if H.shape != (3, 3):
        raise ValueError(f"H_T_to_R must be shape (3,3), got {H.shape}")

    return H


def _shared_stem_from_tif(
    tif_path: Path,
    *,
    reflected_suffix: str = "_ROI-R",
    transmitted_suffix: str = "_ROI-T",
) -> str:
    """
    Return the shared acquisition stem used for Stage 3 output filenames
    by removing the channel-specific suffixes and extension from a TIFF path.
    """
    stem = tif_path.stem
    for suffix in (reflected_suffix, transmitted_suffix):
        suffix = str(suffix)
        if suffix and stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def prepare_stage3_context(
    tif_R_path: str | Path,
    tif_T_path: str | Path,
    H_T_to_R: np.ndarray | str | Path,
    settings,
    *,
    model=None,
    outdir: str | Path | None = None,
    R_stage1_csv: str | Path | None = None,
    T_stage1_csv: str | Path | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    pair_stem: str | None = None,
) -> Stage3RunContext:
    """
    Prepare shared state for one Stage 3 global-fitting run.

    This function performs the setup that is common to free-ratio and fixed-ratio
    global fitting (used in pipeline stage 3):
    - resolve R/T TIFF paths
    - load the T-to-R homography
    - count the overlapping frame range
    - create the output directory
    - load the global dual-channel spline PSF model, unless one was supplied

    It does not create ROI batches, run GPU fitting, write localization CSVs, or
    apply filtering/grouping.
    """

    tif_R_path = Path(tif_R_path)
    tif_T_path = Path(tif_T_path)

    if pair_stem is not None:
        shared_stem = str(pair_stem)
    else:
        shared_stem = _shared_stem_from_tif(
            tif_R_path,
            reflected_suffix=getattr(settings, "reflected_suffix", "_ROI-R"),
            transmitted_suffix=getattr(settings, "transmitted_suffix", "_ROI-T"),
        )
    H = _load_homography_input(H_T_to_R)

    _, nR = count_readable_tiff_prefix(tif_R_path)
    _, nT = count_readable_tiff_prefix(tif_T_path)
    n_min = min(nR, nT)

    # Use only the overlapping frame range shared by both TIFF stacks.
    f0 = int(max(0, start_frame))
    fend = n_min if max_frames is None else min(n_min, f0 + int(max_frames))
    total_stage3_frames = max(0, fend - f0)

    if outdir is None:
        outdir = settings.output_dir()
    else:
        outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    if R_stage1_csv is None:
        R_stage1_csv = outdir / "R_fits.csv"
    else:
        R_stage1_csv = Path(R_stage1_csv)

    if T_stage1_csv is None:
        T_stage1_csv = outdir / "T_fits.csv"
    else:
        T_stage1_csv = Path(T_stage1_csv)

    # Load once so free-ratio and fixed-ratio Stage 3 runs use the same prepared
    # global PSF model.
    if model is None:
        fmt = psf_model_format_normalized(settings)

        if fmt == "SMAP":
            model = load_global_dual_channel_spline_model(
                settings.psf_model,
                zseed_mode="z0",
                verbose=True,
            )

        elif fmt == "uiPSF":
            model = load_uipsf_global_dual_channel_spline_model(
                settings.psf_model,
                coeff_key=settings.uipsf_coeff_key,
                z0_index=settings.uipsf_z0_index,
                normf=settings.uipsf_normf,
                swap_xy_axes=settings.uipsf_swap_xy_axes,
                verbose=True,
            )

        else:
            raise ValueError(f"Unsupported psf_model_format={fmt!r}")

    return Stage3RunContext(
        tif_R_path=tif_R_path,
        tif_T_path=tif_T_path,
        shared_stem=shared_stem,
        H_T_to_R=H,
        outdir=outdir,
        R_stage1_csv=R_stage1_csv,
        T_stage1_csv=T_stage1_csv,
        model=model,
        total_stage3_frames=total_stage3_frames,
    )


def make_stage3_paired_roi_batch_iterator(
    ctx: Stage3RunContext,
    settings,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    on_frame=None,
):
    """
    Create the configured Stage 3 paired-ROI GPU-batch iterator.

    This is a pipeline-level adapter around the lower-level preprocessing iterator.
    It fills in the standard Stage 3 inputs from Stage3RunContext and settings.

    The lower-level iterator reads Stage 1 localization CSVs, transforms T-channel
    localizations into R space, pairs nearby R/T localizations, merges coordinates,
    cuts paired R/T ROI images, and yields GPU-sized batches for global fitting.

    Returns
    -------
    iterator
        Yields tuples that can be converted into Stage3PairedRoiBatch objects.
    """
    return iter_stage3_paired_roi_batches_from_stage1_csvs(
        ctx.tif_R_path,
        ctx.tif_T_path,
        ctx.R_stage1_csv,
        ctx.T_stage1_csv,
        ctx.H_T_to_R,
        offset=settings.offset,
        camera_gain=settings.camera_gain,
        QE=settings.qe,
        roi_size=settings.roi_size,
        batch_size=settings.gpu_batch_size,
        start_frame=start_frame,
        max_frames=max_frames,
        pair_max_linf_dist_px=float(settings.stage3_pair_max_linf_dist_px),
        on_frame=on_frame,
    )


# =============================================================================
# Stage 3 paired-ROI GPU-batch data structures
# =============================================================================
@dataclass(frozen=True)
class Stage3PairedRoiBatch:
    """
    Named container for one Stage 3 paired-ROI GPU batch.

    Each object contains many paired reflected/transmitted ROI cutouts plus the
    metadata needed by the global GPU fitter.

    Array contracts
    ---------------
    rois_R, rois_T
        Image-order ROI cutouts with shape (N, Y, X). They are converted to loclib
        layout later by the GPU helper layer.

    R_centers, T_centers
        Integer ROI centers in each channel, shape (N, 2), stored as [x, y].

    R_dxdy, T_dxdy
        Subpixel residuals from the merged molecular coordinate to each rounded
        channel ROI center, shape (N, 2), stored as [dx, dy].

    zseed_abs
        Absolute spline z-index seeds for the global fit, shape (N,).

    frames
        Frame numbers for each paired localization, shape (N,).

    pair_ids
        Stable pair IDs used to trace Stage 3 rows back to paired Stage 1 detections.

    This is a GPU-batch container, not a folder-level dataset batch.
    """
    rois_R: np.ndarray
    rois_T: np.ndarray
    R_centers: np.ndarray
    T_centers: np.ndarray
    R_dxdy: np.ndarray
    T_dxdy: np.ndarray
    zseed_abs: np.ndarray
    frames: np.ndarray
    pair_ids: np.ndarray

    @property
    def n_pairs(self) -> int:
        if self.rois_R.shape != self.rois_T.shape:
            raise ValueError(
                f"Paired ROI shape mismatch: rois_R={self.rois_R.shape}, "
                f"rois_T={self.rois_T.shape}"
            )
        return int(self.rois_R.shape[0])


def unpack_stage3_paired_roi_batch(batch) -> Stage3PairedRoiBatch:
    """
    Convert one Stage 3 paired-ROI tuple into a named object.

    The preprocessing iterator yields positional tuples for speed and simplicity.
    Stage drivers wrap each tuple in Stage3PairedRoiBatch immediately so downstream
    GPU fitting code can use named fields instead of positional indexing.
    """
    (
        rois_R,
        rois_T,
        R_centers,
        T_centers,
        R_dxdy,
        T_dxdy,
        zseed_abs,
        frames,
        pair_ids,
    ) = batch

    return Stage3PairedRoiBatch(
        rois_R=rois_R,
        rois_T=rois_T,
        R_centers=R_centers,
        T_centers=T_centers,
        R_dxdy=R_dxdy,
        T_dxdy=T_dxdy,
        zseed_abs=zseed_abs,
        frames=frames,
        pair_ids=pair_ids,
    )


# =============================================================================
# CSV path and writing helpers
# =============================================================================
@dataclass(frozen=True)
class Stage3TempCsvPaths:
    """
    Temporary Stage 3 CSV paths used before final row counts are known.

    Final output filenames include localization counts and filtering/grouping
    metadata, so Stage 3 first writes to stable temporary names and renames them
    after fitting is complete.
    """
    full_csv_tmp: Path
    filtered_csv_tmp: Path


def prepare_stage3_temp_csv_paths(
    outdir: Path,
    *,
    mode_name: str,
    out_csv: str | Path | None = None,
) -> Stage3TempCsvPaths:
    """
    Prepare temporary CSV output paths for one global-fitting run. (pipeline stage 3)

    Existing temporary files are deleted at the start of a run so repeated pipeline
    executions do not append to stale outputs.
    """
    if out_csv is None:
        full_csv_tmp = outdir / f"global_{mode_name}_fits_full.csv"
    else:
        full_csv_tmp = Path(out_csv)

    filtered_csv_tmp = outdir / f"global_{mode_name}_fits_filtered_grouped.csv"

    # Start each run from clean temporary CSVs.
    if full_csv_tmp.exists():
        full_csv_tmp.unlink()
    if filtered_csv_tmp.exists():
        filtered_csv_tmp.unlink()

    return Stage3TempCsvPaths(
        full_csv_tmp=full_csv_tmp,
        filtered_csv_tmp=filtered_csv_tmp,
    )


def append_stage3_full_csv(
    df: pd.DataFrame,
    full_csv_tmp: Path,
    settings,
    *,
    header_written: bool,
    rows_written: int,
) -> tuple[bool, int]:
    """
    Append one global fit result dataframe to the full unfiltered CSV. (pipeline stage 3)

    The full CSV is written incrementally during fitting so large runs do not need
    to keep all unfiltered localizations in memory.

    Returns
    -------
    header_written
        Updated flag indicating whether the CSV header has already been written.
    rows_written
        Updated total number of full/unfiltered rows written so far.
    """
    if not getattr(settings, "save_stage3_full_csv", True):
        return header_written, rows_written

    if df is None or df.empty:
        return header_written, rows_written

    df.to_csv(
        full_csv_tmp,
        index=False,
        mode=("a" if header_written else "w"),
        header=not header_written,
    )

    return True, rows_written + int(len(df))


def finalize_stage3_filtered_csv(
    filtered_result_dfs: list[pd.DataFrame],
    filtered_csv_tmp: Path,
    settings,
) -> tuple[int, pd.DataFrame | None]:
    """
    Finalize the filtered/grouped Stage 3 CSV.

    Filtered result dataframes are accumulated in memory, then concatenated once
    after fitting so temporal grouping can be applied across the full fitted frame
    range. The grouped metadata dataframe is written to the temporary filtered CSV.

    Returns
    -------
    total_rows_written_filtered
        Number of rows written to the filtered/grouped CSV.
    df_all
        Final filtered/grouped dataframe, or None if no filtered CSV was written.
    """
    if not getattr(settings, "save_stage3_filtered_csv", True):
        return 0, None

    if not filtered_result_dfs:
        return 0, None

    df_all = pd.concat(filtered_result_dfs, ignore_index=True)
    logger.info("Filtered before grouping: %d", len(df_all))

    from .stage3_filtering_grouping import (
        group_stage3_dataframe,
    )

    df_all = group_stage3_dataframe(df_all, settings)

    if df_all.empty:
        return 0, df_all

    logger.info("Writing filtered/grouped CSV...")
    df_all.to_csv(filtered_csv_tmp, index=False)
    total_rows_written_filtered = int(len(df_all))
    logger.info("Filtered/grouped CSV written: %s", filtered_csv_tmp)

    return total_rows_written_filtered, df_all


def rename_stage3_csv_if_written(
    tmp_csv: Path,
    final_csv: Path,
    *,
    rows_written: int,
    enabled: bool,
    label: str,
) -> Path:
    """
    Rename a temporary Stage 3 CSV after the final row count is known.

    If CSV writing was disabled, or if no rows were written, the temporary path is
    returned unchanged. If renaming fails, the temporary file is kept and a warning
    is logged.
    """
    if not enabled or int(rows_written) <= 0:
        return tmp_csv

    try:
        if final_csv.exists():
            final_csv.unlink()
        tmp_csv.rename(final_csv)
        return final_csv
    except Exception as e:
        logger.warning(
            "Could not rename %s to %s (%s). Keeping temporary name.",
            label,
            final_csv.name,
            e,
        )
        return tmp_csv


class CsvBatchWriter:
    """
    Small buffered CSV writer used for incremental pipeline outputs.

    Dataframes are accumulated in memory and flushed after every `flush_every`
    write calls. This reduces CSV I/O overhead while preserving the same final file
    format.
    """
    def __init__(self, path: Path, flush_every: int = 10):
        self.path = Path(path)
        self.flush_every = int(flush_every)
        self._buffered_dfs: list[pd.DataFrame] = []
        self._header_written = False
        if self.path.exists():
            self.path.unlink()

    def write(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self._buffered_dfs.append(df)
        if len(self._buffered_dfs) >= self.flush_every:
            self._flush()

    def finalize(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._buffered_dfs:
            return

        # Write the header only once, then append subsequent buffered dataframes.
        pd.concat(self._buffered_dfs, ignore_index=True).to_csv(
            self.path,
            index=False,
            mode=("a" if self._header_written else "w"),
            header=not self._header_written,
        )
        self._header_written = True
        self._buffered_dfs.clear()

