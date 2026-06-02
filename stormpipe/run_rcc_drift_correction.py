from pathlib import Path

import pandas as pd

from rcc_drift import RCCDriftConfig, correct_drift_xyz


# ============================================================
# EDIT THESE SETTINGS
# ============================================================

CSV_FILE = Path(
    r"D:\single_signal_test\old_test_data\peakdetection_smapmatch_2026-05-05\peakDetect-SMAPmatchCS4_CS4N2_Acquisition-2_global_fixed_ratio_fits_full_244Klocs_ch1-98K_ch2-147K.csv"
)

FRAME_COL = "frame"
X_COL = "x [nm]"
Y_COL = "y [nm]"
Z_COL = "z [nm]"

OUTPUT_SUFFIX = "_rcc_drift_corrected"


# ============================================================
# DRIFT CORRECTION SETTINGS
# ============================================================

CONFIG = RCCDriftConfig(
    correct_xy=True,
    correct_z=True,

    xy_timepoints=20,
    xy_pixel_size_nm=5.0,
    xy_peak_window_pix=7,

    z_timepoints=20,
    z_bin_width_nm=5.0,
    z_peak_window_pix=9,
    z_range_nm=(-400.0, 400.0),
    z_slice_width_nm=200.0,

    max_drift_nm=1000.0,
    max_reconstruction_size_pix=4096,

    smooth_mode="spline",

    require_min_locs=True,
)


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Could not find CSV file:\n{CSV_FILE.resolve()}")

    print(f"Loading localizations from:\n{CSV_FILE}")
    locs = pd.read_csv(CSV_FILE)

    required_cols = [FRAME_COL, X_COL, Y_COL]

    if CONFIG.correct_z:
        required_cols.append(Z_COL)

    missing = [col for col in required_cols if col not in locs.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + f"\n\nAvailable columns are:\n{list(locs.columns)}"
        )

    locs[FRAME_COL] = locs[FRAME_COL].astype(int)

    print("Running RCC drift correction...")

    corrected, _, _ = correct_drift_xyz(
        locs,
        config=CONFIG,
        frame_col=FRAME_COL,
        x_col=X_COL,
        y_col=Y_COL,
        z_col=Z_COL,
    )

    output_csv = CSV_FILE.with_name(CSV_FILE.stem + OUTPUT_SUFFIX + ".csv")
    corrected.to_csv(output_csv, index=False)

    print("\nDone.")
    print(f"Corrected localizations saved to:\n{output_csv}")


if __name__ == "__main__":
    main()