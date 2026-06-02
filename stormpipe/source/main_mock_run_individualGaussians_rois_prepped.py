# main_gaussian_pipeline.py

import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loclib_ctypes import localizationlib
import dualch_pipelinelib as pipeline


# ===============================
#           SETTINGS
# ===============================
Nz=98
z0=47
dz_nm=20.0
zstarts = pipeline.compute_zstart_index(Nz, dz_nm, z0)
print(f"zstarts: {zstarts}")

SAVE_ROI_IMAGES = True # Toggle for saving ROI images with localization overlay. Set to True for debugging, False for production
GAUSS_FIT_FRAMES = 10000
RUN_PREPROCESSING = False  # Switch to skip ROI detection + Gaussian fitting steps. Set to False to move directly to global fitting step


DATA_DIR = Path(r"D:\mle_fit_TEST\python_test")
OUTPUT_DIR = DATA_DIR / "output"
CHANNEL_MAP = {"R": 0, "T": 1}

OFFSET_ADU = 100.0
CAMERA_GAIN = 0.23
QE = 1.0

MIN_DISTANCE = 7
DOG_SIGMA1 = 1.2
DOG_SIGMA2 = 3.0
THRESHOLD_FACTOR = 1.7

ROI_SIZE = 13
MAX_ITERS = 30
FIXED_SIGMA = 1.0
FREE_SIGMA = False
BATCH_SIZE = 5000
ROIS_PER_FIT = 15000

def find_channel_files(folder: Path) -> dict:
    R_files = sorted(folder.glob("*ROI-R.tif"))
    T_files = sorted(folder.glob("*ROI-T.tif"))
    if len(R_files) != 1 or len(T_files) != 1:
        raise FileNotFoundError(f"Expected 1 ROI-R and 1 ROI-T file in {folder}, found R={len(R_files)} T={len(T_files)}")
    return {"R": str(R_files[0]), "T": str(T_files[0])}

def fit_channel_gauss(tif_path: str, tag: str, out_root: Path, max_frames: int = None) -> tuple[int, int]:
    overlay_cache = [] if SAVE_ROI_IMAGES else None
    print(f"\n=== Processing channel {tag}: {tif_path} ===")
    out_dir = out_root / tag
    vis_dir = out_dir / "rois_with_localizations"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    stack = pipeline.photon_correct(tif_path, offset=OFFSET_ADU, camera_gain=CAMERA_GAIN, QE=QE)
    if max_frames is not None and max_frames < stack.shape[0]:
        print(f"[{tag}] Limiting to first {max_frames} frames (of {stack.shape[0]})")
        stack = stack[:max_frames]


    from dualch_pipelinelib import fit_rois_blockwise

    loc = localizationlib()
    results = []

    def extract_rois_fn(frame, frame_idx):
        filtered = pipeline.noise_correct(frame)
        peaks = pipeline.peak_detect(
            filtered,
            threshold_factor=THRESHOLD_FACTOR,
            sigma1=DOG_SIGMA1,
            sigma2=DOG_SIGMA2,
            min_distance=MIN_DISTANCE,
        )
        return pipeline.extract_rois(frame, peaks, roi_size=ROI_SIZE, frame_index=frame_idx)

    def fit_fn(roi_batch):
        P, CRLB, LL, loc_dict = loc.loc_2Dgauss_mleGPU(
            roi_batch,
            free_sigma=FREE_SIGMA,
            fixed_sigma=FIXED_SIGMA,
            max_nfev=MAX_ITERS
        )
        return P, CRLB, LL, loc_dict

    print("Detecting ROIs and fitting in batches...")
    def on_localization(meta_batch, loc_data):
        dx = loc_data["x"].reshape(-1)
        dy = loc_data["y"].reshape(-1)
        photons = loc_data["photons"].reshape(-1)
        bg = loc_data["bg"].reshape(-1)
        iters = loc_data["iterations"].reshape(-1)
        LL = loc_data["loglikelihood"].reshape(-1) if "loglikelihood" in loc_data else np.zeros_like(dx)

        for j, meta in enumerate(meta_batch):
            x_full = float(meta["offset_x"]) + dx[j]
            y_full = float(meta["offset_y"]) + dy[j]
            results.append({
                "channel": tag,
                "frame": int(meta["frame"]),
                "x_pix": x_full,
                "y_pix": y_full,
                "photons": float(photons[j]),
                "bg_e_per_px": float(bg[j]),
                "x_abs_roi": float(loc_data["x_abs"][j]),
                "y_abs_roi": float(loc_data["y_abs"][j]),
                "dx_roi": float(dx[j]),
                "dy_roi": float(dy[j]),
                "iters": float(iters[j]),
                "LogL": float(LL[j]),
            })

            if SAVE_ROI_IMAGES:
                overlay_cache.append((meta["roi"], loc_data["x_abs"][j], loc_data["y_abs"][j]))


    fit_rois_blockwise(
        image_stack=stack,
        extract_rois_fn=extract_rois_fn,
        fit_fn=fit_fn,
        roisperfit=ROIS_PER_FIT,
        roi_size=ROI_SIZE,
        on_localization=on_localization,  # ✅ correct param name
        show_progress=True
    )



    df = pd.DataFrame(results)
    out_csv = out_dir / f"localizations_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} localizations to {out_csv}")

    if SAVE_ROI_IMAGES and overlay_cache:
        print("Saving ROI overlay images...")
        for i, (roi, u, v) in enumerate(overlay_cache):
            fig, ax = plt.subplots()
            ax.imshow(roi, cmap="gray", interpolation="nearest")
            ax.plot(u, v, "rx", markersize=8, markeredgewidth=1.5)
            ax.axis("off")
            fname = vis_dir / f"roi_{i:05d}.png"
            plt.savefig(fname, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        print(f"Saved {len(overlay_cache)} ROI overlays to {vis_dir}")
    else:
        print("Skipping ROI image saving.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = find_channel_files(DATA_DIR)

    if RUN_PREPROCESSING:
        # Step 1: Fit 10k frames from each channel
        size_r = fit_channel_gauss(files["R"], "R", OUTPUT_DIR, max_frames=GAUSS_FIT_FRAMES)
        size_t = fit_channel_gauss(files["T"], "T", OUTPUT_DIR, max_frames=GAUSS_FIT_FRAMES)

        if size_r != size_t:
            print(f"⚠️ Warning: Image sizes differ: R={size_r}, T={size_t}")

        # Step 2–4: Register channels and extract transform matrix
        pipeline.register_channels(
            path_ref_csv=OUTPUT_DIR / "R" / "localizations_R.csv",
            path_target_csv=OUTPUT_DIR / "T" / "localizations_T.csv",
            save_transformed=True,
            output_dir=str(OUTPUT_DIR)
        )

        from dualch_pipelinelib import load_localizations, apply_homography
        t_csv_path = OUTPUT_DIR / "T" / "localizations_T.csv"
        df_t, coords_t = load_localizations(t_csv_path)
        H = np.load(OUTPUT_DIR / "transform_matrix_projective.npy")
        coords_t_transformed = apply_homography(coords_t, H)
        df_t["x_pix_transformed"] = coords_t_transformed[:, 0]
        df_t["y_pix_transformed"] = coords_t_transformed[:, 1]
        df_t.to_csv(OUTPUT_DIR / "T" / "Transformed_T.csv", index=False)

        print("✅ Channel registration complete.\n✅ Preprocessing complete.")

    else:
        print("⚠️ Skipping preprocessing (ROI detection, fitting, registration)...")

    # === GLOBAL LOCALIZATION PREP ===
    print("\n🔧 Starting global localization ROI prep...")

    from dualch_pipelinelib import (
        photon_correct, noise_correct, peak_detect,
        combine_peaks_dual_channel, process_frames_dual_channel
    )

    # Load full image stacks
    stack_R = photon_correct(files["R"], offset=OFFSET_ADU, camera_gain=CAMERA_GAIN, QE=QE)
    stack_T = photon_correct(files["T"], offset=OFFSET_ADU, camera_gain=CAMERA_GAIN, QE=QE)

    assert stack_R.shape == stack_T.shape, "R and T stacks must have same shape"
    stitched_width = stack_R.shape[2]

    transform_matrix = np.load(OUTPUT_DIR / "transform_matrix_projective.npy")

    # Frame-by-frame peak detect → match → extract ROIs → save overlay
    rois_R, rois_T, roi_meta = process_frames_dual_channel(
        stack_R=stack_R,
        stack_T=stack_T,
        transform_matrix=transform_matrix,
        stitched_width=stitched_width,
        roi_size=ROI_SIZE,
        match_radius=5.0,
        threshold_factor=THRESHOLD_FACTOR,
        sigma1=DOG_SIGMA1,
        sigma2=DOG_SIGMA2,
        min_distance=MIN_DISTANCE,
        output_dir=OUTPUT_DIR,
        combine_peaks_dual_channel=combine_peaks_dual_channel,
        noise_correct=noise_correct,
        peak_detect=peak_detect,
        save_visualizations=SAVE_ROI_IMAGES  # Set False for faster batch runs
    )

    print(f"✅ Global dual-channel peak processing done. {len(roi_meta)} paired ROIs ready.")

    # Optional: save metadata to file for later
    import pandas as pd
    df_meta = pd.DataFrame(roi_meta)
    df_meta.to_csv(OUTPUT_DIR / "combined_peak_metadata.csv", index=False)
    print(f"📝 ROI metadata saved to: combined_peak_metadata.csv")

    print("\n👉 Ready to pass ROIs into GPU fitter (next step)")


    # --- Save dual-channel ROI overlays with weighted-average position ---
    if SAVE_ROI_IMAGES:
        vis_dir = OUTPUT_DIR / "dual" / "rois_with_localizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Heuristics: try common meta keys for the overlay cross
        def _get_xy(meta):
            # Prefer weighted/global initial guess if present; else fall back to per-channel ROI center
            for kx, ky in [
                ("x_abs_roi", "y_abs_roi"),     # same names as your Gauss stage
                ("x_center", "y_center"),       # sometimes returned by pipeline
                ("xw", "yw"),                   # "weighted" naming
                ("x", "y"),                     # generic
            ]:
                if kx in meta and ky in meta:
                    return float(meta[kx]), float(meta[ky])
            # Default: center of ROI
            h, w = rois_R.shape[1:3]
            return (w - 1) / 2.0, (h - 1) / 2.0

        import matplotlib.pyplot as plt

        n = min(len(roi_meta), len(rois_R), len(rois_T))
        for i in range(n):
            roi_r = rois_R[i]
            roi_t = rois_T[i]
            u, v = _get_xy(roi_meta[i])

            # save side-by-side figure
            fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.2))
            for ax, roi, title in zip(axes, [roi_r, roi_t], ["R", "T"]):
                ax.imshow(roi, cmap="gray", interpolation="nearest")
                ax.plot(u, v, "rx", markersize=7, markeredgewidth=1.5)
                ax.set_title(title)
                ax.axis("off")
            out_png = vis_dir / f"roi_{i:05d}.png"
            plt.tight_layout(pad=0.1)
            plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        print(f"Saved {n} dual-channel ROI overlays to: {vis_dir}")



if __name__ == "__main__":
    main()
