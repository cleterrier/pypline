# pipeline_paths.py

from __future__ import annotations

from pathlib import Path

from .pipeline_config import Settings, globloc_mode_tag


def make_unique_output_dir(base_dir: Path) -> Path:
    """
    Create base_dir if it does not exist. If it exists, create base_dir_2,
    base_dir_3, etc.
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir

    i = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{i}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        i += 1


def make_batch_output_root(settings: Settings) -> Path:
    """
    Create a fresh sibling GlobLoc output folder next to the experiment/session
    folder.
    """
    experiment_dir = Path(settings.data_dir)
    mode_tag = globloc_mode_tag(settings)

    base = experiment_dir.parent / f"{experiment_dir.name}_globloc_{mode_tag}"
    return make_unique_output_dir(base)


def make_intermediate_root(settings: Settings) -> Path:
    """
    Create and return the stable folder used for reusable Stage 1/2 intermediates.

    Both flat and recursive input modes use the same convention:

        data_dir / settings.intermediate_dir_name

    This keeps globloc_only runs simple: point data_dir at the same raw data
    folder/session and the pipeline will rediscover the raw TIFFs and reuse the
    matching intermediates.
    """
    root = Path(settings.data_dir) / settings.intermediate_dir_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def localization_registration_paths(
    intermediate_dir: Path,
    pair_stem: str,
) -> dict[str, Path]:
    """
    Pair-specific Stage 1 localization and Stage 2 registration output paths.

    These files are reusable intermediates for GlobLoc-only runs and are stored
    in the stable intermediate folder:

        data_dir / settings.intermediate_dir_name
    """
    intermediate_dir = Path(intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    return {
        "R_stage1_csv": intermediate_dir / f"{pair_stem}_R_fits.csv",
        "T_stage1_csv": intermediate_dir / f"{pair_stem}_T_fits.csv",
        "homography_npy": intermediate_dir / f"{pair_stem}_homography_T_to_R.npy",
        "homography_txt": intermediate_dir / f"{pair_stem}_homography_T_to_R.txt",
        "T_stage1_in_R_csv": intermediate_dir / f"{pair_stem}_T_fits_in_Rframe.csv",
    }