"""Summarize SelfRDB LiQA test predictions into CSV and Markdown reports."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASET_DIR = r"E:\tmp\selfrdb_liqa_t1_ged4_256"
DEFAULT_LOG_DIR = r"logs\liqa_t1_ged4\version_0\test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create slice-level and case-level reports for LiQA SelfRDB runs."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--target", default="GED4")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--pred-path",
        default=None,
        help="Optional explicit path to pred.npy. Defaults to <log-dir>/test_samples/pred.npy.",
    )
    return parser.parse_args()


def import_runtime_dependencies():
    try:
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio
        from skimage.metrics import structural_similarity
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing reporting dependencies. Run inside the project conda "
            "environment, for example: conda activate selfrdb"
        ) from exc

    return np, peak_signal_noise_ratio, structural_similarity


def slice_number(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_stack(directory: Path, np):
    files = sorted(directory.glob("slice_*.npy"), key=slice_number)
    if not files:
        raise FileNotFoundError(f"No slice_*.npy files found in {directory}")
    return np.stack([np.load(file).astype("float32") for file in files]), files


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_pred_path(args: argparse.Namespace) -> Path:
    if args.pred_path:
        pred_path = Path(args.pred_path)
    else:
        log_dir = Path(args.log_dir)
        pred_path = log_dir / "test_samples" / "pred.npy"
        if not pred_path.exists():
            pred_path = log_dir / "pred.npy"

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    return pred_path


def normalize_prediction_shape(pred, np):
    pred = np.asarray(pred)
    if pred.ndim == 2:
        return pred[None, ...]
    if pred.ndim == 3:
        return pred
    if pred.ndim == 4 and pred.shape[1] == 1:
        return pred[:, 0, ...]
    if pred.ndim == 4 and pred.shape[-1] == 1:
        return pred[..., 0]
    raise ValueError(f"Unsupported prediction shape: {pred.shape}")


def mean_norm(array, np):
    array = np.abs(array)
    denom = array.mean(axis=(-1, -2), keepdims=True)
    return np.divide(array, denom, out=np.zeros_like(array), where=denom > 1e-8)


def center_crop(array, shape):
    h, w = array.shape[-2:]
    crop_h, crop_w = shape
    top = h // 2 - crop_h // 2
    left = w // 2 - crop_w // 2
    return array[..., top : top + crop_h, left : left + crop_w]


def compute_slice_metrics(gt, pred, mask, np, psnr_fn, ssim_fn):
    gt = gt.squeeze() if gt.ndim == 4 else gt
    pred = pred.squeeze() if pred.ndim == 4 else pred
    if gt.ndim == 2:
        gt = gt[None, ...]
    if pred.ndim == 2:
        pred = pred[None, ...]
    if gt.shape != pred.shape:
        raise ValueError(f"Target and prediction shapes differ: {gt.shape} vs {pred.shape}")

    if mask is not None:
        if mask.shape[-2:] != gt.shape[-2:]:
            gt = center_crop(gt, mask.shape[-2:])
            pred = center_crop(pred, mask.shape[-2:])
        gt = mean_norm(gt * mask, np)
        pred = mean_norm(pred * mask, np)
    else:
        gt = mean_norm(gt, np)
        pred = mean_norm(pred, np)

    psnrs = []
    ssims = []
    for gt_slice, pred_slice in zip(gt, pred):
        data_range = float(np.nanmax(gt_slice))
        if not np.isfinite(data_range) or data_range <= 0:
            psnrs.append(float("nan"))
            ssims.append(float("nan"))
            continue
        psnrs.append(float(psnr_fn(gt_slice, pred_slice, data_range=data_range)))
        ssims.append(float(ssim_fn(gt_slice, pred_slice, data_range=data_range) * 100.0))

    return np.asarray(psnrs), np.asarray(ssims)


def mean_std(values, np) -> tuple[float, float]:
    values = np.asarray(values, dtype="float64")
    return float(np.nanmean(values)), float(np.nanstd(values))


def write_slice_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "output_index",
        "slice_file",
        "case_id",
        "vendor",
        "slice_index",
        "foreground_ratio",
        "psnr",
        "ssim",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_case_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["case_id", "vendor", "num_slices", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    path: Path,
    dataset_dir: Path,
    pred_path: Path,
    num_slices: int,
    num_cases: int,
    slice_summary: tuple[float, float, float, float],
    case_summary: tuple[float, float, float, float],
    case_rows: list[dict[str, str]],
) -> None:
    slice_psnr_mean, slice_psnr_std, slice_ssim_mean, slice_ssim_std = slice_summary
    case_psnr_mean, case_psnr_std, case_ssim_mean, case_ssim_std = case_summary

    lines = [
        "# LiQA SelfRDB T1->GED4 Report",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- Predictions: `{pred_path}`",
        f"- Test cases: {num_cases}",
        f"- Test slices: {num_slices}",
        "",
        "## Primary Metrics (Case Averaged)",
        "",
        f"- PSNR: {case_psnr_mean:.2f} +/- {case_psnr_std:.2f}",
        f"- SSIM: {case_ssim_mean:.2f} +/- {case_ssim_std:.2f}",
        "",
        "## Slice Metrics",
        "",
        f"- PSNR: {slice_psnr_mean:.2f} +/- {slice_psnr_std:.2f}",
        f"- SSIM: {slice_ssim_mean:.2f} +/- {slice_ssim_std:.2f}",
        "",
        "## Case Table",
        "",
        "| Case | Vendor | Slices | PSNR | SSIM |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for row in case_rows:
        lines.append(
            "| {case_id} | {vendor} | {num_slices} | {psnr_mean} | {ssim_mean} |".format(
                **row
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np, psnr_fn, ssim_fn = import_runtime_dependencies()

    dataset_dir = Path(args.dataset_dir)
    metadata_path = Path(args.metadata) if args.metadata else dataset_dir / "metadata_test.csv"
    pred_path = resolve_pred_path(args)
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    target, target_files = load_stack(dataset_dir / args.target / "test", np)
    mask_dir = dataset_dir / "mask" / "test"
    mask = None
    if mask_dir.exists():
        mask, _ = load_stack(mask_dir, np)
    pred = normalize_prediction_shape(np.load(pred_path).astype("float32"), np)

    if len(pred) != len(target):
        raise ValueError(
            f"Prediction count differs from target count: {len(pred)} vs {len(target)}"
        )

    metadata = read_metadata(metadata_path)
    if len(metadata) != len(target):
        raise ValueError(
            f"Metadata rows differ from target count: {len(metadata)} vs {len(target)}"
        )

    psnrs, ssims = compute_slice_metrics(target, pred, mask, np, psnr_fn, ssim_fn)

    slice_rows = []
    grouped = defaultdict(list)
    for i, row in enumerate(metadata):
        case_id = row.get("case_id", "unknown")
        vendor = row.get("vendor", "unknown")
        slice_row = {
            "output_index": row.get("output_index", str(i)),
            "slice_file": row.get("slice_file", target_files[i].name),
            "case_id": case_id,
            "vendor": vendor,
            "slice_index": row.get("slice_index", ""),
            "foreground_ratio": row.get("foreground_ratio", ""),
            "psnr": f"{psnrs[i]:.6f}",
            "ssim": f"{ssims[i]:.6f}",
        }
        slice_rows.append(slice_row)
        grouped[case_id].append((vendor, psnrs[i], ssims[i]))

    case_rows = []
    case_psnr_means = []
    case_ssim_means = []
    for case_id in sorted(grouped):
        values = grouped[case_id]
        vendor = values[0][0]
        case_psnrs = [value[1] for value in values]
        case_ssims = [value[2] for value in values]
        psnr_mean, psnr_std = mean_std(case_psnrs, np)
        ssim_mean, ssim_std = mean_std(case_ssims, np)
        case_psnr_means.append(psnr_mean)
        case_ssim_means.append(ssim_mean)
        case_rows.append(
            {
                "case_id": case_id,
                "vendor": vendor,
                "num_slices": str(len(values)),
                "psnr_mean": f"{psnr_mean:.6f}",
                "psnr_std": f"{psnr_std:.6f}",
                "ssim_mean": f"{ssim_mean:.6f}",
                "ssim_std": f"{ssim_std:.6f}",
            }
        )

    slice_psnr_mean, slice_psnr_std = mean_std(psnrs, np)
    slice_ssim_mean, slice_ssim_std = mean_std(ssims, np)
    case_psnr_mean, case_psnr_std = mean_std(case_psnr_means, np)
    case_ssim_mean, case_ssim_std = mean_std(case_ssim_means, np)

    write_slice_metrics(output_dir / "slice_metrics.csv", slice_rows)
    write_case_metrics(output_dir / "case_metrics.csv", case_rows)
    write_markdown_report(
        output_dir / "report.md",
        dataset_dir,
        pred_path,
        len(target),
        len(case_rows),
        (slice_psnr_mean, slice_psnr_std, slice_ssim_mean, slice_ssim_std),
        (case_psnr_mean, case_psnr_std, case_ssim_mean, case_ssim_std),
        case_rows,
    )

    print(f"Case-averaged PSNR: {case_psnr_mean:.2f} +/- {case_psnr_std:.2f}")
    print(f"Case-averaged SSIM: {case_ssim_mean:.2f} +/- {case_ssim_std:.2f}")
    print(f"Wrote reports to: {output_dir}")


if __name__ == "__main__":
    main()
