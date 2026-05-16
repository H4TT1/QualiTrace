import json
import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def _png_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.png") if path.is_file())


def _validate_image(path: Path, expected_channels: int) -> dict:
    issue = {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "mode": None,
        "width": None,
        "height": None,
        "issues": [],
    }

    if issue["size_bytes"] == 0:
        issue["issues"].append("empty_file")
        return issue

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            issue["mode"] = img.mode
            issue["width"], issue["height"] = img.size
            channels = len(img.getbands())
    except (OSError, UnidentifiedImageError) as exc:
        issue["issues"].append(f"corrupted_or_unreadable:{type(exc).__name__}")
        return issue

    if channels != expected_channels or issue["mode"] != "RGB":
        issue["issues"].append(f"not_rgb:{issue['mode']}")

    return issue


def validate_mvtec_dataset(
    data_dir: str,
    category: str,
    expected_channels: int = 3,
    min_train_test_ratio: float | None = None,
    max_train_test_ratio: float | None = None,
) -> dict:
    category_root = Path(data_dir) / category
    train_root = category_root / "train" / "good"
    test_root = category_root / "test"

    report = {
        "data_dir": data_dir,
        "category": category,
        "train_dir": str(train_root),
        "test_dir": str(test_root),
        "train_count": 0,
        "test_count": 0,
        "test_counts_by_class": {},
        "train_test_ratio": None,
        "issues": [],
        "invalid_images": [],
        "passed": True,
    }

    if not category_root.exists():
        report["issues"].append(f"missing_category_dir:{category_root}")
    if not train_root.exists():
        report["issues"].append(f"missing_train_good_dir:{train_root}")
    if not test_root.exists():
        report["issues"].append(f"missing_test_dir:{test_root}")

    train_images = _png_files(train_root)
    test_images = _png_files(test_root)
    report["train_count"] = len(train_images)
    report["test_count"] = len(test_images)

    if report["train_count"] == 0:
        report["issues"].append("empty_train_split")
    if report["test_count"] == 0:
        report["issues"].append("empty_test_split")

    if report["test_count"]:
        for subdir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            report["test_counts_by_class"][subdir.name] = len(_png_files(subdir))

    if report["test_count"]:
        ratio = report["train_count"] / report["test_count"]
        report["train_test_ratio"] = ratio
        if min_train_test_ratio is not None and ratio < min_train_test_ratio:
            report["issues"].append(f"train_test_ratio_below_min:{ratio:.4f}")
        if max_train_test_ratio is not None and ratio > max_train_test_ratio:
            report["issues"].append(f"train_test_ratio_above_max:{ratio:.4f}")

    for image_path in train_images + test_images:
        image_report = _validate_image(image_path, expected_channels)
        if image_report["issues"]:
            report["invalid_images"].append(image_report)

    report["passed"] = not report["issues"] and not report["invalid_images"]
    return report


def save_validation_report(report: dict, output_dir: str, filename: str = "data_validation_report.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, filename)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report_path


def raise_if_validation_failed(report: dict):
    if report["passed"]:
        return

    issue_preview = report["issues"][:5]
    invalid_preview = [item["path"] for item in report["invalid_images"][:5]]
    raise ValueError(
        "Dataset validation failed. "
        f"Issues: {issue_preview}. "
        f"Invalid images: {invalid_preview}. "
        "See data_validation_report.json for details."
    )
