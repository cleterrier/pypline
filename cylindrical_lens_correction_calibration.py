# cylindrical_lens_correction_calibration.py AUTHOR: Christopher Parperis 2026

"""
User entry point for cylindrical-lens field-distortion homography calibration.
"""

from __future__ import annotations

import logging
from pathlib import Path

from stormpipe.cylindrical_lens_homography_calibration import (
    CylindricalLensCalibrationSettings,
    run_cylindrical_lens_homography_calibration,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


settings = CylindricalLensCalibrationSettings(
    # =========================================================================
    # Required paths
    # =========================================================================
    calibration_dir=Path(r"G:\3Dto2D_correction_example_folder"),
    output_dir=Path(r"G:\3Dto2D_correction_example_folder\cylindrical_lens_calibration_output"),
    psf_model_format="uiPSF",
    psf_model=Path(
        r"Z:\DATA\NC_DATA_ABBELIGHT_#1\Christopher\241121_axcal\mat_for_uiPSF\uiPSF_output\xySwap_PSFmodel_zernike_vector_multi.h5"
    ),

    # =========================================================================
    # Calibration pair naming
    # =========================================================================
    # Example expected pairs:
    #   round001_cylindrical_lens.tif
    #   round001_no_cylindrical_lens.tif
    cylindrical_suffix="_cylindrical_lens_ROI-R",
    no_cylindrical_suffix="_no_cylindrical_lens_ROI-R",
    tiff_extension=".tif",
    input_search_mode="flat",  # "flat" or "recursive"

    # =========================================================================
    # Channel/model choice
    # =========================================================================
    # R or T as reference.
    # (Set as R or T, depending on 'main' channel for the global fitting 
    # pipeline. This will fit the cylindrical lens images of the beads with the
    # spline PSF model unique to that channel.)
    channel_tag="R",

    # =========================================================================
    # Camera parameters
    # =========================================================================
    pixelsize_nm=97.0,
    offset=100.0,
    camera_gain=0.23,
    qe=1.0,

    # =========================================================================
    # Detection / ROI extraction
    # =========================================================================
    roi_size=13,
    thr_factor_cylindrical=2.0,
    thr_factor_no_cylindrical=2.0,
    sigma1=1.2,
    sigma2=3.2,
    min_distance=5,
    keep_peak_winner=True,

    # None = use all readable frames.
    max_frames_per_movie=None,

    # =========================================================================
    # GPU fitting
    # =========================================================================
    usecuda=1,
    gpu_iterations=150,
    gpu_batch_size=9000,
    em_excess_noise=1,
    ri_mismatch=1.0,

    # No-cylindrical-lens Gaussian fit.
    gaussian_free_sigma=True,
    gaussian_fixed_sigma_px=1.5,
    gaussian_iterations=100,

    # =========================================================================
    # Bead centroiding
    # =========================================================================
    cluster_eps_px=1.5,
    min_detections_per_bead=20,
    centroid_method="median",  # "median" or "weighted_mean"

    # Optional filters before clustering. Start with None, then tighten after QC.
    centroid_max_locprec_nm=None,
    centroid_min_photons=None,
    require_converged_for_centroids=True,

    # =========================================================================
    # Bead pairing / homography
    # =========================================================================
    max_centroid_pair_dist_px=20.0,
    ransac_reproj_thresh_px=2.0,
    min_pairs_for_homography=4,

    # =========================================================================
    # Run behavior
    # =========================================================================
    overwrite_existing_locs=False,
    skip_failed_pairs=True,
)


def main() -> None:
    result = run_cylindrical_lens_homography_calibration(settings)

    print("\nCylindrical-lens homography calibration complete.")
    print(f"H_cyl_to_nocyl TXT: {result['H_txt']}")
    print(f"H_cyl_to_nocyl NPY: {result['H_npy']}")
    print(f"Summary CSV:        {result['homography_summary_csv']}")
    print(f"Correspondences:    {result['global_bead_correspondences_csv']}")


if __name__ == "__main__":
    main()