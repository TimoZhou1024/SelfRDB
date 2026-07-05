"""Prepare LiQA NIfTI volumes for SelfRDB's NumpyDataset layout."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_TRAIN_LIST = r"E:\liverGAN\data\train.txt"
DEFAULT_VAL_LIST = r"E:\liverGAN\data\val.txt"
DEFAULT_TEST_LIST = r"E:\liverGAN\data\test.txt"
DEFAULT_OUTPUT = r"E:\tmp\selfrdb_liqa_t1_ged4_256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LiQA case lists into the modality/split/slice_*.npy "
            "directory structure expected by SelfRDB."
        )
    )
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-list", default=DEFAULT_VAL_LIST)
    parser.add_argument("--test-list", default=DEFAULT_TEST_LIST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default="T1")
    parser.add_argument("--target", default="GED4")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--foreground-min-ratio", type=float, default=0.01)
    parser.add_argument("--clip-lower", type=float, default=1.0)
    parser.add_argument("--clip-upper", type=float, default=99.5)
    parser.add_argument(
        "--max-cases-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit; by default all cases are converted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing non-empty output directory before writing.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip cases with missing source/target files instead of failing.",
    )
    return parser.parse_args()


def import_runtime_dependencies():
    try:
        import numpy as np
        import nibabel as nib
        from scipy.ndimage import affine_transform
        from skimage.transform import resize
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing preprocessing dependencies. Run inside the project conda "
            "environment, for example: conda activate selfrdb"
        ) from exc

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = None

    return np, nib, affine_transform, resize, tqdm


def read_case_list(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Case list not found: {path}")

    cases = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError(f"Case list is empty: {path}")

    return cases


def prepare_output_dir(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output}. "
                "Use --overwrite to regenerate it."
            )

        resolved = output.resolve()
        if resolved.parent == resolved or len(resolved.parts) < 3:
            raise ValueError(f"Refusing to overwrite unsafe output path: {output}")
        shutil.rmtree(resolved)

    output.mkdir(parents=True, exist_ok=True)


def resolve_modality_file(case_dir: Path, modality: str) -> Path:
    return case_dir / f"{modality}.nii.gz"


def resample_to_reference(
    moving_array,
    moving_affine,
    reference_shape,
    reference_affine,
    affine_transform,
    np,
):
    """Resample moving image data onto the reference voxel grid."""
    transform = np.linalg.inv(moving_affine).dot(reference_affine)
    matrix = transform[:3, :3]
    offset = transform[:3, 3]
    return affine_transform(
        moving_array,
        matrix=matrix,
        offset=offset,
        output_shape=reference_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype("float32")


def ensure_3d(array, case_id: str, modality: str, np):
    array = np.asarray(array)
    if array.ndim == 4 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(
            f"Expected 3D volume for {case_id} {modality}, got shape {array.shape}"
        )
    return array


def load_volume(path: Path, nib, np):
    image = nib.load(str(path))
    array = np.asarray(image.dataobj, dtype=np.float32)
    return array, image.affine


def to_slice_first(array, case_id: str, modality: str, np):
    array = ensure_3d(array, case_id, modality, np)
    return np.moveaxis(array, -1, 0)


def normalize_volume(array, lower: float, upper: float, np):
    array = array.astype("float32", copy=False)
    finite = np.isfinite(array)
    foreground = finite & (array != 0)
    values = array[foreground]
    if values.size == 0:
        values = array[finite]
    if values.size == 0:
        return np.zeros_like(array, dtype="float32")

    low, high = np.percentile(values, [lower, upper])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = float(values.max())
        low = float(values.min())
    if high <= low:
        return np.zeros_like(array, dtype="float32")

    array = np.clip(array, low, high)
    array = (array - low) / (high - low)
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0).astype("float32")


def resize_slice(array2d, image_size: int, resize, is_mask: bool = False):
    array2d = array2d.astype("float32", copy=False)
    if array2d.shape == (image_size, image_size):
        return array2d if not is_mask else (array2d > 0.5).astype("uint8")

    resized_array = resize(
        array2d,
        (image_size, image_size),
        order=0 if is_mask else 1,
        mode="constant",
        cval=0.0,
        clip=True,
        preserve_range=True,
        anti_aliasing=not is_mask,
    )
    if is_mask:
        return (resized_array > 0.5).astype("uint8")
    return resized_array.astype("float32")


def write_subject_ids(path: Path, subject_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for subject_id in subject_ids:
            f.write(f"- {subject_id}\n")


def process_split(
    split: str,
    case_dirs: list[Path],
    args: argparse.Namespace,
    output: Path,
    np,
    nib,
    affine_transform,
    resize,
    tqdm,
) -> tuple[int, list[str]]:
    source_dir = output / args.source / split
    target_dir = output / args.target / split
    mask_dir = output / "mask" / split
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    if split == "test":
        mask_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output / f"metadata_{split}.csv"
    subject_ids = []
    rows = []
    slice_out_idx = 0
    iterator = tqdm(case_dirs, desc=f"Preparing {split}") if tqdm else case_dirs

    for case_dir in iterator:
        case_id = case_dir.name
        vendor = case_dir.parent.name
        source_path = resolve_modality_file(case_dir, args.source)
        target_path = resolve_modality_file(case_dir, args.target)

        if not source_path.exists() or not target_path.exists():
            message = (
                f"Missing required files for {case_id}: "
                f"{source_path.name} exists={source_path.exists()}, "
                f"{target_path.name} exists={target_path.exists()}"
            )
            if args.skip_missing:
                print(f"Skipping {message}")
                continue
            raise FileNotFoundError(message)

        source_data, source_affine = load_volume(source_path, nib, np)
        target_data, target_affine = load_volume(target_path, nib, np)
        source_data = ensure_3d(source_data, case_id, args.source, np)
        target_data = ensure_3d(target_data, case_id, args.target, np)
        source_shape_original = source_data.shape

        source_resampled = resample_to_reference(
            moving_array=source_data,
            moving_affine=source_affine,
            reference_shape=target_data.shape,
            reference_affine=target_affine,
            affine_transform=affine_transform,
            np=np,
        )

        source_array = to_slice_first(source_resampled, case_id, args.source, np)
        target_array = to_slice_first(target_data, case_id, args.target, np)
        if source_array.shape != target_array.shape:
            raise ValueError(
                f"Resampled source and target shapes differ for {case_id}: "
                f"{source_array.shape} vs {target_array.shape}"
            )

        source_array = normalize_volume(
            source_array, args.clip_lower, args.clip_upper, np
        )
        target_raw = target_array.astype("float32", copy=False)
        target_mask = np.isfinite(target_raw) & (target_raw != 0)
        target_array = normalize_volume(
            target_raw, args.clip_lower, args.clip_upper, np
        )

        for z_index in range(target_array.shape[0]):
            mask_slice = target_mask[z_index]
            foreground_ratio = float(mask_slice.mean())
            if foreground_ratio < args.foreground_min_ratio:
                continue

            slice_name = f"slice_{slice_out_idx}.npy"
            source_slice = resize_slice(
                source_array[z_index], args.image_size, resize
            )
            target_slice = resize_slice(
                target_array[z_index], args.image_size, resize
            )

            np.save(source_dir / slice_name, source_slice.astype("float32"))
            np.save(target_dir / slice_name, target_slice.astype("float32"))

            if split == "test":
                resized_mask = resize_slice(
                    mask_slice.astype("float32"),
                    args.image_size,
                    resize,
                    is_mask=True,
                )
                np.save(mask_dir / slice_name, resized_mask.astype("uint8"))
                subject_ids.append(case_id)

            rows.append(
                {
                    "split": split,
                    "output_index": slice_out_idx,
                    "slice_file": slice_name,
                    "case_id": case_id,
                    "vendor": vendor,
                    "case_path": str(case_dir),
                    "source_modality": args.source,
                    "target_modality": args.target,
                    "source_file": str(source_path),
                    "target_file": str(target_path),
                    "slice_index": z_index,
                    "foreground_ratio": f"{foreground_ratio:.8f}",
                    "source_shape_original": "x".join(map(str, source_shape_original)),
                    "target_shape": "x".join(map(str, target_array.shape)),
                }
            )
            slice_out_idx += 1

    if slice_out_idx == 0:
        raise RuntimeError(f"No slices were written for split '{split}'")

    fieldnames = [
        "split",
        "output_index",
        "slice_file",
        "case_id",
        "vendor",
        "case_path",
        "source_modality",
        "target_modality",
        "source_file",
        "target_file",
        "slice_index",
        "foreground_ratio",
        "source_shape_original",
        "target_shape",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{split}: wrote {slice_out_idx} paired slices")
    return slice_out_idx, subject_ids


def main() -> None:
    args = parse_args()
    np, nib, affine_transform, resize, tqdm = import_runtime_dependencies()

    output = Path(args.output)
    prepare_output_dir(output, args.overwrite)

    split_lists = {
        "train": Path(args.train_list),
        "val": Path(args.val_list),
        "test": Path(args.test_list),
    }
    split_cases = {split: read_case_list(path) for split, path in split_lists.items()}
    if args.max_cases_per_split is not None:
        if args.max_cases_per_split <= 0:
            raise ValueError("--max-cases-per-split must be positive")
        split_cases = {
            split: cases[: args.max_cases_per_split]
            for split, cases in split_cases.items()
        }

    summary_rows = []
    test_subject_ids = []
    for split in ("train", "val", "test"):
        count, subject_ids = process_split(
            split,
            split_cases[split],
            args,
            output,
            np,
            nib,
            affine_transform,
            resize,
            tqdm,
        )
        summary_rows.append(
            {
                "split": split,
                "cases": len(split_cases[split]),
                "slices": count,
                "source": args.source,
                "target": args.target,
                "image_size": args.image_size,
                "foreground_min_ratio": args.foreground_min_ratio,
            }
        )
        if split == "test":
            test_subject_ids = subject_ids

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "cases",
                "slices",
                "source",
                "target",
                "image_size",
                "foreground_min_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_subject_ids(output / "subject_ids.yaml", test_subject_ids)
    print(f"Done. SelfRDB dataset written to: {output}")


if __name__ == "__main__":
    main()
