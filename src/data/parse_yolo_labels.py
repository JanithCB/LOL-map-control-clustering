# src/data/parse_yolo_labels.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

LOGGER = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "image_id",
    "label_file",
    "class_id",
    "class_name",
    "x_center_norm",
    "y_center_norm",
    "width_norm",
    "height_norm",
    "x_min_norm",
    "y_min_norm",
    "x_max_norm",
    "y_max_norm",
    "x_center_px",
    "y_center_px",
    "width_px",
    "height_px",
    "image_width",
    "image_height",
]


def _empty_labels_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected schema."""
    return pd.DataFrame(columns=EXPECTED_COLUMNS)


def read_class_names(names_path: str | Path) -> dict[int, str]:
    """
    Read YOLO class names from a .names file.

    Parameters
    ----------
    names_path : str | Path
        Path to a file containing one class name per line.

    Returns
    -------
    dict[int, str]
        Mapping of class_id to class_name.
    """
    path = Path(names_path)
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")

    class_map: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if not name:
                continue
            class_map[idx] = name

    return class_map


def _get_image_size(image_path: str | Path | None) -> tuple[int | None, int | None]:
    """Return (width, height) for an image, or (None, None) if unavailable."""
    if image_path is None:
        return None, None

    path = Path(image_path)
    if not path.exists():
        LOGGER.warning("Image file not found: %s", path)
        return None, None

    try:
        with Image.open(path) as img:
            width, height = img.size
        return width, height
    except Exception as exc:
        LOGGER.warning("Failed to read image size for %s: %s", path, exc)
        return None, None


def _is_valid_norm(value: float) -> bool:
    """Check if a normalized coordinate/value is within [0, 1]."""
    return 0.0 <= value <= 1.0


def _parse_label_line(
    line: str,
    line_number: int,
    label_path: Path,
    image_id: str,
    class_map: dict[int, str] | None,
    image_width: int | None,
    image_height: int | None,
) -> dict[str, Any] | None:
    """
    Parse a single YOLO label line into a dictionary row.

    Returns None if the row is malformed or invalid.
    """
    parts = line.strip().split()
    if not parts:
        return None

    if len(parts) != 5:
        LOGGER.warning(
            "Skipping malformed row in %s at line %d: expected 5 values, got %d",
            label_path,
            line_number,
            len(parts),
        )
        return None

    try:
        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except ValueError:
        LOGGER.warning(
            "Skipping malformed numeric row in %s at line %d: %s",
            label_path,
            line_number,
            line.strip(),
        )
        return None

    if not all(_is_valid_norm(v) for v in (x_center, y_center, width, height)):
        LOGGER.warning(
            "Skipping out-of-range normalized row in %s at line %d: %s",
            label_path,
            line_number,
            line.strip(),
        )
        return None

    if width <= 0 or height <= 0:
        LOGGER.warning(
            "Skipping non-positive bbox size row in %s at line %d: %s",
            label_path,
            line_number,
            line.strip(),
        )
        return None

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    if not all(_is_valid_norm(v) for v in (x_min, y_min, x_max, y_max)):
        LOGGER.warning(
            "Skipping bbox extending outside normalized bounds in %s at line %d: %s",
            label_path,
            line_number,
            line.strip(),
        )
        return None

    class_name = class_map.get(class_id, str(class_id)) if class_map else str(class_id)

    if image_width is not None and image_height is not None:
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height
    else:
        x_center_px = None
        y_center_px = None
        width_px = None
        height_px = None

    return {
        "image_id": image_id,
        "label_file": str(label_path),
        "class_id": class_id,
        "class_name": class_name,
        "x_center_norm": x_center,
        "y_center_norm": y_center,
        "width_norm": width,
        "height_norm": height,
        "x_min_norm": x_min,
        "y_min_norm": y_min,
        "x_max_norm": x_max,
        "y_max_norm": y_max,
        "x_center_px": x_center_px,
        "y_center_px": y_center_px,
        "width_px": width_px,
        "height_px": height_px,
        "image_width": image_width,
        "image_height": image_height,
    }


def parse_label_file(
    label_path: str | Path,
    image_path: str | Path | None = None,
    class_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Parse a single YOLO label file into a DataFrame.

    Parameters
    ----------
    label_path : str | Path
        Path to the YOLO label file.
    image_path : str | Path | None, optional
        Optional path to the corresponding image. If provided, pixel coordinates
        and image dimensions are computed using Pillow.
    class_map : dict[int, str] | None, optional
        Optional mapping from class_id to class_name.

    Returns
    -------
    pd.DataFrame
        One row per bounding box with normalized and pixel-space coordinates.
        Empty label files return an empty DataFrame with the correct columns.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    image_id = label_path.stem
    image_width, image_height = _get_image_size(image_path)

    try:
        raw_text = label_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        LOGGER.warning("Failed to read label file %s: %s", label_path, exc)
        return _empty_labels_df()

    if not raw_text:
        return _empty_labels_df()

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        parsed = _parse_label_line(
            line=line,
            line_number=line_number,
            label_path=label_path,
            image_id=image_id,
            class_map=class_map,
            image_width=image_width,
            image_height=image_height,
        )
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        return _empty_labels_df()

    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def _find_matching_image(
    label_path: Path,
    images_dir: Path | None,
    image_extensions: tuple[str, ...],
) -> Path | None:
    """Find an image matching a label file stem in the given images directory."""
    if images_dir is None:
        return None

    for ext in image_extensions:
        candidate = images_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def parse_dataset(
    labels_dir: str | Path,
    images_dir: str | Path | None = None,
    names_path: str | Path | None = None,
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> pd.DataFrame:
    """
    Parse a full YOLO dataset directory into a single DataFrame.

    Parameters
    ----------
    labels_dir : str | Path
        Directory containing YOLO .txt label files.
    images_dir : str | Path | None, optional
        Directory containing matching image files.
    names_path : str | Path | None, optional
        Path to a .names file for class names.
    image_extensions : tuple[str, ...], optional
        Allowed image extensions for matching label stems.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all parsed labels with an added bbox_area_norm column.
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels path is not a directory: {labels_dir}")

    images_path = Path(images_dir) if images_dir is not None else None
    class_map = read_class_names(names_path) if names_path is not None else None

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        LOGGER.warning("No label files found in %s", labels_dir)
        df = _empty_labels_df()
        df["bbox_area_norm"] = pd.Series(dtype=float)
        return df

    parsed_frames: list[pd.DataFrame] = []
    images_with_no_labels = 0

    for label_file in label_files:
        image_path = _find_matching_image(label_file, images_path, image_extensions)

        if images_path is not None and image_path is None:
            LOGGER.warning(
                "No matching image found for label file: %s (searched in %s)",
                label_file.name,
                images_path,
            )

        df_one = parse_label_file(
            label_path=label_file,
            image_path=image_path,
            class_map=class_map,
        )

        if df_one.empty:
            images_with_no_labels += 1

        parsed_frames.append(df_one)

    if parsed_frames:
        df = pd.concat(parsed_frames, ignore_index=True)
    else:
        df = _empty_labels_df()

    if df.empty:
        df = _empty_labels_df()
        df["bbox_area_norm"] = pd.Series(dtype=float)
    else:
        df["bbox_area_norm"] = df["width_norm"] * df["height_norm"]

    df.attrs["images_with_no_labels"] = images_with_no_labels
    return df


def save_parsed_labels(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save parsed labels to CSV, creating parent directories if necessary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved parsed labels to %s", output_path)


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return summary statistics for a parsed YOLO labels DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed labels DataFrame.

    Returns
    -------
    dict[str, Any]
        Summary statistics including row/image/class counts.
    """
    if df.empty:
        return {
            "n_rows": 0,
            "n_images": 0,
            "n_classes": 0,
            "class_counts": {},
            "images_with_no_labels": df.attrs.get("images_with_no_labels", 0),
        }

    class_counts = (
        df["class_name"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    summary = {
        "n_rows": int(len(df)),
        "n_images": int(df["image_id"].nunique()),
        "n_classes": int(df["class_id"].nunique()),
        "class_counts": class_counts,
        "images_with_no_labels": int(df.attrs.get("images_with_no_labels", 0)),
    }
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Parse YOLO-format minimap label files into a CSV table."
    )
    parser.add_argument("--labels-dir", required=True, help="Directory of YOLO label .txt files.")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional directory of corresponding image files.",
    )
    parser.add_argument(
        "--names-path",
        default=None,
        help="Optional path to .names class mapping file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save parsed CSV output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    df = parse_dataset(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        names_path=args.names_path,
    )
    save_parsed_labels(df, args.output)

    summary = dataset_summary(df)
    print("Parsed YOLO dataset summary:")
    print(f"  rows: {summary['n_rows']}")
    print(f"  images: {summary['n_images']}")
    print(f"  classes: {summary['n_classes']}")
    print(f"  images_with_no_labels: {summary['images_with_no_labels']}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()