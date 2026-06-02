# uiPSF_prepare_bead_stacks.py AUTHOR: Christopher Parperis 2026
#Stitch R/T bead stacks together for input to ui-PSF

from pathlib import Path
import h5py
import tifffile as tif
import numpy as np


# ============================================================
# Set this to your parent directory containing many child folders of z-stacks. Each child folder should contain one *_cam_R.tif and one *_cam_T.tif, which will be combined side-by-side and saved to the output folder.
# ============================================================
parent_dir = Path(r"g:\pypline\calibration\241121_axcal")

# Output folder for uiPSF-ready .mat files
output_dir = parent_dir / "mat_for_uiPSF"
output_dir.mkdir(exist_ok=True)

if not parent_dir.exists():
    raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

child_dirs = sorted([p for p in parent_dir.iterdir() if p.is_dir()])

if not child_dirs:
    raise FileNotFoundError(f"No child folders found in: {parent_dir}")

print(f"Found {len(child_dirs)} child folder(s).")
print(f"Output folder: {output_dir}")

processed = 0
skipped = 0

for child in child_dirs:
    # Skip the output directory itself if rerunning the script
    if child.resolve() == output_dir.resolve():
        continue

    print(f"\n--- Folder: {child.name} ---")

    r_files = sorted(child.glob("*_cam_R.tif"))
    t_files = sorted(child.glob("*_cam_T.tif"))

    if len(r_files) == 0:
        print("Skipping: no *_cam_R.tif found.")
        skipped += 1
        continue

    if len(t_files) == 0:
        print("Skipping: no *_cam_T.tif found.")
        skipped += 1
        continue

    if len(r_files) > 1 or len(t_files) > 1:
        print("Skipping: expected exactly one R file and one T file.")
        print(f"  Found {len(r_files)} R file(s), {len(t_files)} T file(s).")
        skipped += 1
        continue

    r_path = r_files[0]
    t_path = t_files[0]

    R = tif.imread(r_path)
    T = tif.imread(t_path)

    print(f"R: {r_path.name}, shape={R.shape}, dtype={R.dtype}")
    print(f"T: {t_path.name}, shape={T.shape}, dtype={T.dtype}")

    if R.shape != T.shape:
        print("Skipping: R/T shape mismatch.")
        print(f"  R shape: {R.shape}")
        print(f"  T shape: {T.shape}")
        skipped += 1
        continue

    if R.ndim != 3:
        print(f"Skipping: expected 3D stack [z, y, x], got {R.shape}")
        skipped += 1
        continue

    # uiPSF converts to float32 internally anyway.
    # Keeping float32 reduces .mat file size.
    R = R.astype(np.float32, copy=False)
    T = T.astype(np.float32, copy=False)

    out_path = output_dir / f"{child.name}_cam_RT.mat"

    # uiPSF reads .mat files using h5py and treats each non-metadata key
    # as a channel when channeltype == "multi".
    # channel0 = R, channel1 = T.
    with h5py.File(out_path, "w") as f:
        f.create_dataset("channel0", data=R, compression="gzip")
        f.create_dataset("channel1", data=T, compression="gzip")

        # Optional metadata; uiPSF's loader ignores a key named "metadata".
        meta = f.create_group("metadata")
        meta.attrs["channel0"] = "R"
        meta.attrs["channel1"] = "T"
        meta.attrs["source_R"] = r_path.name
        meta.attrs["source_T"] = t_path.name
        meta.attrs["assumed_axis_order"] = "z, y, x"

    print(f"Saved: {out_path.name}")
    processed += 1

print("\n========================================")
print(f"Done. Processed: {processed}, Skipped: {skipped}")
print(f"Output folder: {output_dir}")
print("========================================")