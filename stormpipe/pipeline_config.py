# pipeline_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Settings:
    # =========================================================================
    # 1. IO, calibration, and run mode
    # =========================================================================

    # Folder containing raw reflected/transmitted TIFF files.
    data_dir: Path = Path(r"D:\batch_test\data_folder_test\bz_xy")

    # Path to the 2-channel spline-interpolated PSF calibration model.
    psf_model: Path = Path(r"D:\single_signal_test\Axcal_inputZStack_cam_R_3dcal.mat")

    # PSF model file format:
    #   "SMAP"  -> MATLAB/SMAP .mat model with SXY/SXY_g fields
    #   "uiPSF" -> Python uiPSF .h5 model with locres/coeff
    #
    # psf_model remains the path to the model file.
    psf_model_format: str = "SMAP"  # "SMAP" or "uiPSF"

    # uiPSF coefficient dataset under /locres.
    # First implementation assumes this dataset is already stored in the correct
    # orientation by uiPSF calibration:
    #     locres/<uipsf_coeff_key>.shape == (2, 64, Z, Y, X)
    uipsf_coeff_key: str = "coeff"

    # Optional uiPSF z0 override. If None, infer z0 as Z // 2.
    uipsf_z0_index: int | None = None

    # If True, swap the final two spatial axes of the uiPSF coefficient tensor:
    #     (C, 64, Z, Y, X) -> (C, 64, Z, X, Y)
    #
    # This is a loader-level rescue option for uiPSF models trained with the
    # wrong xy convention. It does not swap image ROIs, coordinates, homographies,
    # or dTS offsets.
    uipsf_swap_xy_axes: bool = False

    # Pipeline mode:
    #   "full"         -> run Stage 1 + Stage 2 + Stage 3 + drift correction
    #   "globloc_only" -> reuse existing Stage 1/2 intermediates and run Stage 3 onward
    pipeline_mode: str = "full"  # "full" or "globloc_only"

    # Calibration channel mapping inside the PSF model.
    # Usually R/reflected is channel 0 and T/transmitted is channel 1.
    channel_map: Dict[str, int] = field(default_factory=lambda: {"R": 0, "T": 1})

    # =========================================================================
    # 2. Input file naming
    # =========================================================================
    #
    # These describe the raw TIFF naming convention used for reflected/transmitted
    # file-pair discovery and shared-stem output naming.

    reflected_suffix: str = "ROI-R"
    transmitted_suffix: str = "ROI-T"
    tiff_extension: str = ".tif"

    # Input discovery mode:
    #   "flat"      -> search only data_dir (i.e. all paired R+T tiffs are in one folder together)
    #   "recursive" -> search data_dir and all subfolders for paired R+T tiffs
    input_search_mode: str = "flat"

    # Minimum readable frame count required in each channel for processing.
    # If less than this, the reflected/transmitted pair is skipped.
    # Use 0 to disable filtering. Set to e.g. 10_000 to skip snapshot acquisitions.
    min_readable_frames_per_channel: int = 10_000

    # Folder under data_dir where reusable Stage 1/2 intermediates are stored.
    intermediate_dir_name: str = "_globloc_intermediates"

    # =========================================================================
    # 3. Preview
    # =========================================================================

    # If True, display peak-detection previews instead of running full fitting.
    preview_detections: bool = False

    # Frame index used for preview mode.
    preview_frame_index: int = 12332

    # =========================================================================
    # 4. Camera geometry and photon correction
    # =========================================================================

    # Effective pixel size in nm after magnification.
    pixelsize_nm: float = 97

    # Camera voltage offset in ADU / black level.
    offset: float = 100.0

    # Gain value used for converting ADU to photons.
    camera_gain: float = 0.23

    # Quantum efficiency at the experiment wavelength.
    qe: float = 1.0

    # =========================================================================
    # 5. Detection and ROI extraction
    # =========================================================================

    # Odd side length of square ROI cutouts.
    roi_size: int = 13

    # Detection threshold above background for reflected-channel peak detections.
    thr_factor_reflected: float = 1.7

    # Detection threshold above background for transmitted-channel peak detections.
    thr_factor_transmitted: float = 1.7

    # Gaussian smoothing sigma for peak detection, first Gaussian in DoG.
    sigma1: float = 1.2

    # Gaussian smoothing sigma for peak detection, second Gaussian in DoG.
    sigma2: float = 3.2

    # Minimum allowed distance between same-channel peak detections.
    min_distance: int = 5

    # If True, keep the strongest peak within each min_distance neighborhood.
    # If False, reject clustered peaks.
    keep_peak_winner: bool = True

    # =========================================================================
    # 6. GPU fitting
    # =========================================================================

    # Number of GPU MLE iterations.
    gpu_iterations: int = 150

    # EMCCD excess-noise setting passed to fitting/formatting.
    em_excess_noise: int = 1

    # Refractive-index mismatch scaling applied to z positions/errors.
    ri_mismatch: float = 1.0

    # Number of ROIs accumulated before flushing one GPU fitting batch.
    gpu_batch_size: int = 9000

    # =========================================================================
    # 7. Stage 2 channel registration / homography estimation
    # =========================================================================

    # Frame range for registration of Stage 1 localizations.
    # The output homography maps transmitted-channel coordinates into reflected
    # channel coordinates for the dual-channel global fit.
    registration_start_frame: int = 5000

    # End frame is exclusive.
    # If this value is too large for the readable shared frame range, the
    # registration window will be ignored and all frames will be used instead.
    registration_end_frame: int | None = 15000

    # =========================================================================
    # 8. Stage 3 global dual-channel fitting / spectral demixing
    # =========================================================================

    # Maximum L-infinity distance in pixels for pairing Stage 1 R/T localizations
    # before the global dual-channel fit.
    stage3_pair_max_linf_dist_px: float = 5.0

    # If False, stop after Stage 1/2 intermediates.
    run_stage3_global_fit: bool = True

    # Global dual-channel fit mode:
    #   "free_ratio"    -> fit R/T photon counts independently
    #   "fixed_ratios"  -> fit one candidate fixed ratio at a time and choose by LL
    global_fit_mode: str = "fixed_ratios"  # "free_ratio" or "fixed_ratios"

    # Expected photon fractions T / (R + T), for each fluorophore.
    # Values must be between 0 and 1.
    # These are converted internally to T/R for fitting.
    fixed_ratios: tuple[float, ...] = (0.3, 0.65)

    # Filter out fixed-ratio fit localizations where channel assignment is not clear.
    fixed_ratio_reject_ambiguous: bool = True

    # Reject near-ties in best/second LL for the channel assignment.
    # Smaller = stricter.
    fixed_ratio_ll_ratio_threshold: float = 0.999

    # Optional frame window for Stage 3.
    stage3_start_frame: int = 0

    # None = full overlapping R/T stack.
    stage3_max_frames: int | None = None

    # Stage 3 output conventions.
    # Use the brighter channel as the "main" channel for most accurate photon
    # reporting in the global dual-channel fit.
    main_channel: str = "R"  # "R", "T", or "mean"

    # Default channel for free-ratio localizations before spectral demixing.
    # Fixed-ratio mode outputs channels 1..N for N fixed ratios.
    free_ratio_output_channel: int = 0

    # =========================================================================
    # 9. Stage 3 filtering and grouping
    # =========================================================================

    # Save unfiltered Stage 3 global-fit CSV.
    save_stage3_full_csv: bool = True

    # Save filtered/grouped Stage 3 CSV.
    save_stage3_filtered_csv: bool = True

    # Maximum allowed xy localization uncertainty in nm. (applied before drift correction)
    stage3_uncertainty_xy_threshold_nm: float = 30.0

    # If True, reject fits whose final xy position moved too far from the seed.
    stage3_convergence_xy_filter: bool = True

    # Maximum allowed xy displacement from R seed in pixels for convergence filter.
    stage3_conv_xy_threshold_px: float = 3.0

    # If True, require the GPU fit to converge before the iteration cap.
    stage3_require_converged: bool = True

    # If True, assign temporal/spatial blink groups after Stage 3 filtering.
    stage3_grouping_enabled: bool = True

    # Maximum xy distance in pixels for temporal grouping.
    stage3_group_dx_px: float = 1.0

    # Number of dark/missing frames allowed while tracking a group.
    stage3_group_dt_frames: int = 1

    # =========================================================================
    # 10. RCC drift correction
    # =========================================================================

    # Run redundant cross-correlation drift correction.
    run_rcc_drift_correction: bool = True

    # Correct xy drift using RCC.
    rcc_correct_xy: bool = True

    # Correct z drift using RCC.
    rcc_correct_z: bool = True

    # Number of timepoints/windows for xy RCC drift estimation.
    rcc_xy_timepoints: int = 20

    # Rendering pixel size for xy RCC drift estimation.
    rcc_xy_pixel_size_nm: float = 5.0

    # Number of timepoints/windows for z RCC drift estimation.
    rcc_z_timepoints: int = 20

    # Z histogram bin width for z RCC drift estimation.
    rcc_z_bin_width_nm: float = 5.0

    # Maximum allowed drift magnitude during RCC peak search.
    rcc_max_drift_nm: float = 1000.0

    # Z range used for z RCC histograms.
    rcc_z_range_nm: tuple[float, float] = (-400.0, 400.0)

    # Do not crash the whole pipeline/batch if RCC cannot be estimated.
    rcc_skip_on_failure: bool = True

    # =========================================================================
    # 11. COMET drift correction
    # =========================================================================

    # Run COMET drift correction.
    # If both RCC and COMET are enabled, the order is Stage 3 -> RCC -> COMET.
    run_comet_drift_correction: bool = True

    # Suppress COMET stdout/stderr/warnings where possible.
    comet_suppress_output: bool = True

    # COMET segmentation mode.
    comet_segmentation_mode: int = 2

    # COMET segmentation variable.
    comet_segmentation_var: int = 60

    # COMET initial Gaussian sigma in nm.
    comet_initial_sigma_nm: float = 100.0

    # COMET target Gaussian sigma in nm.
    comet_target_sigma_nm: float = 1.0

    # Maximum drift allowed during COMET correction.
    comet_max_drift_nm: float = 300.0

    # COMET boxcar smoothing width.
    comet_boxcar_width: int = 1

    # COMET interpolation method.
    comet_interpolation_method: str = "cubic"

    # Optional cap on localizations per COMET segment.
    comet_max_locs_per_segment: int | None = None

    # Do not crash the whole pipeline/batch if COMET cannot be estimated.
    comet_skip_on_failure: bool = True

    # =========================================================================
    # 12. Render-ready channel CSV export
    # =========================================================================

    # Combine grouped/blinking localizations only for the final render-ready
    # channel CSV exports, after drift correction has already been applied.
    combine_grouped_localizations_for_render: bool = False

    # Apply a calibrated cylindrical-lens xy homography correction only to the final
    # ThunderSTORM/ImageJ render-ready channel exports.
    #
    # This is applied after drift correction and after optional grouped-localization
    # combination, immediately before the render-ready export schema is created.
    apply_cylindrical_lens_xy_correction_for_render: bool = False

    # Path to H_cyl_to_nocyl saved by the cylindrical-lens calibration workflow.
    # Supports .npy or .txt homography files.
    #
    # This placeholder path is ignored unless
    # apply_cylindrical_lens_xy_correction_for_render=True.
    cylindrical_lens_xy_homography: Path = Path(
        r"D:\path\to\cylindrical_lens_calibration_output\H_cyl_to_nocyl.txt"
    )

    # Maximum number of detections allowed in one temporal group for final
    # render-ready exports. (Applied after drift correction, immediately before
    # grouped-localization combination and ThunderSTORM/ImageJ export.)
    render_max_number_in_group: int | None = 5

    # If True, final channel-split CSVs are converted to a compact schema for
    # ThunderSTORM/ImageJ rendering workflows.
    render_ready_channel_csvs: bool = True

    # Add this offset to frame numbers in render-ready exports.
    # 1 gives ImageJ/ThunderSTORM-style one-indexed frames.
    render_frame_offset: int = 1

    # Sign convention for exported z values in render-ready files.
    render_z_sign: float = -1.0

    # Scale factor applied to z values in render-ready files.
    render_z_scale: float = 0.8

    # Scale factor to control xy Gaussians rendered on localizations in final image.
    render_uncertainty_xy_scale: float = 1 / np.sqrt(2) # (Baddeley et al. 2010, Visualization of Localization Microscopy Data)

    # Scale factor to control z uncertainty exported for rendering.
    render_uncertainty_z_scale: float = 1 / np.sqrt(2)

    def thr_factor_for_channel(self, tag: str) -> float:
        if tag == "R":
            return float(self.thr_factor_reflected)
        if tag == "T":
            return float(self.thr_factor_transmitted)
        raise ValueError(f"Unknown channel tag: {tag!r}")

    def __post_init__(self):
        # ---------------------------------------------------------------------
        # IO / naming / run mode validation
        # ---------------------------------------------------------------------
        if self.pipeline_mode not in {"full", "globloc_only"}:
            raise ValueError("pipeline_mode must be 'full' or 'globloc_only'")

        if not self.reflected_suffix:
            raise ValueError("reflected_suffix must be a non-empty string")
        if not self.transmitted_suffix:
            raise ValueError("transmitted_suffix must be a non-empty string")
        if self.reflected_suffix == self.transmitted_suffix:
            raise ValueError("reflected_suffix and transmitted_suffix must be different")
        if not self.tiff_extension.startswith(".") or len(self.tiff_extension) < 2:
            raise ValueError("tiff_extension must start with '.', e.g. '.tif'")
        if self.input_search_mode not in {"flat", "recursive"}:
            raise ValueError("input_search_mode must be 'flat' or 'recursive'")
        if self.min_readable_frames_per_channel < 0:
            raise ValueError("min_readable_frames_per_channel must be >= 0")
        if not self.intermediate_dir_name:
            raise ValueError("intermediate_dir_name must be a non-empty string")
        if Path(self.intermediate_dir_name).is_absolute():
            raise ValueError("intermediate_dir_name must be a relative folder name")

        # ---------------------------------------------------------------------
        # PSF model format validation
        # ---------------------------------------------------------------------
        fmt = str(self.psf_model_format).strip().lower()
        if fmt not in {"smap", "uipsf"}:
            raise ValueError("psf_model_format must be 'SMAP' or 'uiPSF'")

        if fmt == "uipsf":
            if not str(self.uipsf_coeff_key).strip():
                raise ValueError("uipsf_coeff_key must be a non-empty string")

            allowed_uipsf_coeff_keys = {"coeff", "coeff_reverse", "coeff_bead"}
            if self.uipsf_coeff_key not in allowed_uipsf_coeff_keys:
                raise ValueError(
                    "uipsf_coeff_key must be one of "
                    f"{sorted(allowed_uipsf_coeff_keys)}, got {self.uipsf_coeff_key!r}"
                )

            if self.uipsf_z0_index is not None and int(self.uipsf_z0_index) < 0:
                raise ValueError("uipsf_z0_index must be None or >= 0")

            if not isinstance(self.uipsf_swap_xy_axes, bool):
                raise ValueError("uipsf_swap_xy_axes must be True or False")
            
        # ---------------------------------------------------------------------
        # Detection / ROI validation
        # ---------------------------------------------------------------------
        if self.roi_size % 2 == 0:
            raise ValueError(f"roi_size must be odd, got {self.roi_size}")

        if self.thr_factor_reflected <= 0:
            raise ValueError("thr_factor_reflected must be > 0")
        if self.thr_factor_transmitted <= 0:
            raise ValueError("thr_factor_transmitted must be > 0")

        # ---------------------------------------------------------------------
        # Stage 2 registration validation
        # ---------------------------------------------------------------------
        if self.registration_start_frame < 0:
            raise ValueError("registration_start_frame must be >= 0")

        if self.registration_end_frame is not None:
            if self.registration_end_frame <= self.registration_start_frame:
                raise ValueError(
                    "registration_end_frame must be greater than registration_start_frame, "
                    "or None to use all available frames"
                )

        # ---------------------------------------------------------------------
        # Stage 3 fitting / spectral-demixing validation
        # ---------------------------------------------------------------------
        if self.stage3_pair_max_linf_dist_px <= 0:
            raise ValueError("stage3_pair_max_linf_dist_px must be > 0")

        if self.global_fit_mode not in {"free_ratio", "fixed_ratios"}:
            raise ValueError(
                f"Unsupported global_fit_mode={self.global_fit_mode!r}. "
                "Currently implemented: 'free_ratio' or 'fixed_ratios'."
            )

        if self.global_fit_mode == "fixed_ratios":
            if len(self.fixed_ratios) == 0:
                raise ValueError(
                    "fixed_ratios must contain at least one ratio when "
                    "global_fit_mode='fixed_ratios'"
                )
            arr = np.asarray(self.fixed_ratios, dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                raise ValueError("fixed_ratios must all be finite")
            if np.any(arr <= 0) or np.any(arr >= 1):
                raise ValueError(
                    "fixed_ratios must be transmitted/total photon fractions, "
                    "T/(R+T), with values between 0 and 1"
                )

        if self.main_channel not in {"R", "T", "mean"}:
            raise ValueError("main_channel must be 'R', 'T', or 'mean'")

        if self.fixed_ratio_ll_ratio_threshold <= 0:
            raise ValueError("fixed_ratio_ll_ratio_threshold must be > 0")
        if self.fixed_ratio_ll_ratio_threshold > 1.0:
            raise ValueError("fixed_ratio_ll_ratio_threshold must be <= 1.0")

        # ---------------------------------------------------------------------
        # Stage 3 filtering / grouping validation
        # ---------------------------------------------------------------------
        if self.stage3_uncertainty_xy_threshold_nm <= 0:
            raise ValueError("stage3_uncertainty_xy_threshold_nm must be > 0")
        if self.stage3_conv_xy_threshold_px <= 0:
            raise ValueError("stage3_conv_xy_threshold_px must be > 0")
        if self.stage3_group_dx_px <= 0:
            raise ValueError("stage3_group_dx_px must be > 0")
        if self.stage3_group_dt_frames < 0:
            raise ValueError("stage3_group_dt_frames must be >= 0")
        if (
            self.render_max_number_in_group is not None
            and self.render_max_number_in_group < 1
        ):
            raise ValueError("render_max_number_in_group must be None or >= 1")

        # ---------------------------------------------------------------------
        # RCC drift-correction validation
        # ---------------------------------------------------------------------
        if self.rcc_xy_timepoints < 2:
            raise ValueError("rcc_xy_timepoints must be >= 2")
        if self.rcc_z_timepoints < 2:
            raise ValueError("rcc_z_timepoints must be >= 2")
        if self.rcc_xy_pixel_size_nm <= 0:
            raise ValueError("rcc_xy_pixel_size_nm must be > 0")
        if self.rcc_z_bin_width_nm <= 0:
            raise ValueError("rcc_z_bin_width_nm must be > 0")
        if self.rcc_max_drift_nm <= 0:
            raise ValueError("rcc_max_drift_nm must be > 0")
        if len(self.rcc_z_range_nm) != 2:
            raise ValueError("rcc_z_range_nm must be a 2-tuple")
        if self.rcc_z_range_nm[0] >= self.rcc_z_range_nm[1]:
            raise ValueError("rcc_z_range_nm must be increasing, e.g. (-400, 400)")

        # ---------------------------------------------------------------------
        # COMET drift-correction validation
        # ---------------------------------------------------------------------
        if self.comet_segmentation_mode not in {0, 1, 2}:
            raise ValueError("comet_segmentation_mode must be 0, 1, or 2")
        if self.comet_segmentation_var <= 0:
            raise ValueError("comet_segmentation_var must be > 0")
        if self.comet_initial_sigma_nm <= 0:
            raise ValueError("comet_initial_sigma_nm must be > 0")
        if self.comet_target_sigma_nm <= 0:
            raise ValueError("comet_target_sigma_nm must be > 0")
        if self.comet_max_drift_nm <= 0:
            raise ValueError("comet_max_drift_nm must be > 0")
        if self.comet_boxcar_width < 1:
            raise ValueError("comet_boxcar_width must be >= 1")
        if self.comet_interpolation_method not in {"cubic", "catmull-rom"}:
            raise ValueError("comet_interpolation_method must be 'cubic' or 'catmull-rom'")
        if self.comet_max_locs_per_segment is not None and self.comet_max_locs_per_segment <= 0:
            raise ValueError("comet_max_locs_per_segment must be None or > 0")

        # ---------------------------------------------------------------------
        # Render-ready export validation
        # ---------------------------------------------------------------------
        if self.apply_cylindrical_lens_xy_correction_for_render:
            h_path = Path(self.cylindrical_lens_xy_homography)

            if not h_path.exists():
                raise FileNotFoundError(
                    "apply_cylindrical_lens_xy_correction_for_render=True, but "
                    f"cylindrical_lens_xy_homography does not exist: {h_path}"
                )

            if h_path.suffix.lower() not in {".npy", ".txt"}:
                raise ValueError(
                    "cylindrical_lens_xy_homography must be a .npy or .txt homography file"
                )

        if self.render_frame_offset not in {0, 1}:
            raise ValueError("render_frame_offset must be 0 or 1")
        if self.render_z_sign not in {-1.0, 1.0}:
            raise ValueError("render_z_sign must be -1.0 or 1.0")
        if self.render_z_scale <= 0:
            raise ValueError("render_z_scale must be > 0")
        if self.render_uncertainty_xy_scale <= 0:
            raise ValueError("render_uncertainty_xy_scale must be > 0")
        if self.render_uncertainty_z_scale <= 0:
            raise ValueError("render_uncertainty_z_scale must be > 0")

def psf_model_format_normalized(settings: Settings) -> str:
    """
    Return normalized PSF model format tag.

    Keeps Settings user-facing values flexible while giving loader dispatchers
    one canonical spelling.
    """
    fmt = str(getattr(settings, "psf_model_format", "SMAP")).strip().lower()

    if fmt == "smap":
        return "SMAP"

    if fmt == "uipsf":
        return "uiPSF"

    raise ValueError(f"Unsupported psf_model_format={fmt!r}")

def ratio_to_filename_tag(r: float) -> str:
    """
    Convert a ratio into a filename-safe compact tag.

    Examples
    --------
    0.425 -> 0425
    1.860 -> 186
    0.1   -> 01
    1.65  -> 165
    3.889 -> 3889
    """
    s = f"{float(r):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "").replace("-", "m")


def globloc_mode_tag(settings: Settings) -> str:
    if settings.global_fit_mode == "free_ratio":
        return "freeratios"

    if settings.global_fit_mode == "fixed_ratios":
        ratio_tag = "-".join(ratio_to_filename_tag(r) for r in settings.fixed_ratios)
        return f"fixedratios_{ratio_tag}"

    raise ValueError(f"Unsupported global_fit_mode={settings.global_fit_mode!r}")

def channel_split_channels_for_settings(settings: Settings) -> tuple[int, ...]:
    """
    Channels to export for ThunderSTORM/ImageJ.

    free_ratio normally outputs channel 0.
    fixed_ratios outputs channel 1..N.
    """
    if settings.global_fit_mode == "free_ratio":
        return (int(settings.free_ratio_output_channel),)

    if settings.global_fit_mode == "fixed_ratios":
        return tuple(range(1, len(settings.fixed_ratios) + 1))

    return (0, 1, 2, 3, 4)

def fixed_ratios_as_t_over_r(settings: Settings) -> tuple[float, ...]:
    """
    Convert user-facing fixed ratios from T/(R+T) to the T/R convention required
    by the fixed-ratio global fitter.
    """
    ratios = np.asarray(settings.fixed_ratios, dtype=np.float64)

    if not np.all(np.isfinite(ratios)):
        raise ValueError("fixed_ratios must all be finite")
    if np.any(ratios <= 0) or np.any(ratios >= 1):
        raise ValueError(
            "fixed_ratios must be transmitted/total photon fractions, "
            "T/(R+T), with values between 0 and 1"
        )

    ratios_t_over_r = ratios / (1.0 - ratios)
    return tuple(float(r) for r in ratios_t_over_r)