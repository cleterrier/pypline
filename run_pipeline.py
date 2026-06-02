# run_pipeline.py

"""
Main entrypoint for the spectral-demixing GlobLoc STORM pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from stormpipe.pipeline_config import Settings
from stormpipe.workflow_orchestration import preview_batch_detections, run_batch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# =============================================================================
# User run configuration
# =============================================================================

RUN_PREVIEW = False   # If True, runs preview_batch_detections. If False, runs full batch with run_batch.

settings = Settings(
    # =========================================================================
    # Data path + PSF model selection
    # =========================================================================
    data_dir=Path(r"g:\pypline\data"),

    # PSF model format:
    #   "SMAP"  -> calibrated in SMAP module "calibrate3DsplinePSF", saved as .mat file
    #   "uiPSF" -> calibrated with Python "uiPSF" .h5 file
    psf_model_format="uiPSF",

    # Set this path to either:
    #   - SMAP .mat model if psf_model_format="SMAP"
    #   - uiPSF .h5 model if psf_model_format="uiPSF"
    psf_model=Path(r"g:\pypline\calibration\241121_axcal\mat_for_uiPSF\uiPSF_output\xySwap_PSFmodel_zernike_vector_multi.h5"),

    # uiPSF-only options. Ignored when psf_model_format="SMAP".
    uipsf_coeff_key="coeff",       # "coeff", "coeff_reverse", or "coeff_bead"
    uipsf_z0_index=None,           # None -> infer Z // 2
    uipsf_normf=(1.0, 0.4853629229415472),        # neutral channel normalization
    uipsf_swap_xy_axes=False,      # swap final PSF coefficient axes Y/X if needed

    # =========================================================================
    # Input layout
    # =========================================================================
    # "flat":
    #     data_dir contains pairs directly:
    #     a_ROI-R.tif, a_ROI-T.tif, b_ROI-R.tif, b_ROI-T.tif, etc.
    #
    # "recursive":
    #     data_dir is a session/day folder. The pipeline searches subfolders for:
    #         ROI-R.tif / ROI-T.tif
    #
    # Stage 1/2 files (reuseable for stage3 'globloc_only' fitting if desired)
    # are written to:
    #     data_dir / "_globloc_intermediates"
    input_search_mode="recursive",  # "flat" or "recursive"

    # =========================================================================
    # Run mode
    # =========================================================================
    # "full":
    #     run Stage 1 + Stage 2 + Stage 3 + drift correction
    #
    # "globloc_only":
    #     reuse data_dir/_globloc_intermediates and rerun Stage 3 onward
    pipeline_mode="full",  # "full" or "globloc_only"

    # =========================================================================
    # Preview
    # =========================================================================
    preview_frame_index=12332, # Show this frame if preview mode 'True'

    # =========================================================================
    # Camera parameters
    # =========================================================================
    pixelsize_nm=97, # Effective pixel size in nm (i.e. image size after magnification, not the physical pixel size on the chip)
    offset=100.0,                  # Camera voltage offset in ADU (black level)
    camera_gain=0.23,                # Gain value for converting ADU to photons
    qe=1.0, # Quantum efficiency (0 to 1) of the camera at wavelength used in experiment

    # =========================================================================
    # Detection
    # =========================================================================
    thr_factor_reflected=2.5, # Detection threshold above background for peak detections
    thr_factor_transmitted=1.7,
    min_distance=5, # Minimum allowed distance between same-channel peak detections

    # =========================================================================
    # Spectral demixing
    # =========================================================================
    global_fit_mode="fixed_ratios",  # "free_ratio" or "fixed_ratios"

    # If fixed ratio mode, specify as photons transmitted fractions:
    #     T / (R + T)
    fixed_ratios=(0.3, 0.65),

    # =========================================================================
    # Drift correction
    # =========================================================================
    run_rcc_drift_correction=True, # Redundant cross-correlation drift correction
    run_comet_drift_correction=True, # COMET drift correction (runs after RCC if both "True")

    # =========================================================================
    # Render-ready channel CSV export
    # =========================================================================
    apply_cylindrical_lens_xy_correction_for_render=True,  # xy homography to correct cylindrical lens distortion
    cylindrical_lens_xy_homography=Path(
        r"g:\pypline\calibration\3Dto2D_correction_example_folder\cylindrical_lens_calibration_output\H_cyl_to_nocyl.txt"
    ),
)


def main() -> None:
    if RUN_PREVIEW:
        preview_batch_detections(settings)
    else:
        run_batch(settings)


if __name__ == "__main__":
    main()