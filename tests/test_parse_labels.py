# tests/test_parse_labels.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.parse_yolo_labels import (
    EXPECTED_COLUMNS,
    parse_dataset,
    parse_label_file,
    read_class_names,
)


def _make_image(path: Path, size: tuple[int, int] = (100, 200), color=(255, 255, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)


def test_read_class_names(tmp_path: Path) -> None:
    names_file = tmp_path / "mid.names"
    names_file.write_text("Ahri\nLux\nZed\n", encoding="utf-8")

    class_map = read_class_names(names_file)

    assert class_map == {0: "Ahri", 1: "Lux", 2: "Zed"}


def test_parse_valid_yolo_label_file(tmp_path: Path) -> None:
    label_file = tmp_path / "sample.txt"
    label_file.write_text("1 0.50 0.25 0.20 0.10\n", encoding="utf-8")

    class_map = {0: "Ahri", 1: "Lux"}
    df = parse_label_file(label_file, class_map=class_map)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == EXPECTED_COLUMNS

    row = df.iloc[0]
    assert row["image_id"] == "sample"
    assert row["class_id"] == 1
    assert row["class_name"] == "Lux"
    assert row["x_center_norm"] == pytest.approx(0.50)
    assert row["y_center_norm"] == pytest.approx(0.25)
    assert row["width_norm"] == pytest.approx(0.20)
    assert row["height_norm"] == pytest.approx(0.10)
    assert row["x_min_norm"] == pytest.approx(0.40)
    assert row["y_min_norm"] == pytest.approx(0.20)
    assert row["x_max_norm"] == pytest.approx(0.60)
    assert row["y_max_norm"] == pytest.approx(0.30)


def test_parse_label_file_computes_pixel_coordinates(tmp_path: Path) -> None:
    label_file = tmp_path / "frame_001.txt"
    image_file = tmp_path / "frame_001.png"

    label_file.write_text("0 0.50 0.25 0.20 0.10\n", encoding="utf-8")
    _make_image(image_file, size=(100, 200))

    df = parse_label_file(label_file, image_path=image_file, class_map={0: "Ahri"})
    row = df.iloc[0]

    assert row["image_width"] == 100
    assert row["image_height"] == 200
    assert row["x_center_px"] == pytest.approx(50.0)
    assert row["y_center_px"] == pytest.approx(50.0)
    assert row["width_px"] == pytest.approx(20.0)
    assert row["height_px"] == pytest.approx(20.0)


def test_empty_label_file_returns_empty_dataframe_with_expected_columns(tmp_path: Path) -> None:
    label_file = tmp_path / "empty.txt"
    label_file.write_text("", encoding="utf-8")

    df = parse_label_file(label_file)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == EXPECTED_COLUMNS


def test_malformed_lines_are_skipped_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    label_file = tmp_path / "bad_rows.txt"
    label_file.write_text(
        "\n".join(
            [
                "0 0.50 0.50 0.20 0.10",   # valid
                "bad row here",            # malformed
                "1 1.50 0.50 0.20 0.10",   # out of range
                "2 0.40 0.40 -0.10 0.10",  # negative width
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        df = parse_label_file(label_file, class_map={0: "Ahri", 1: "Lux", 2: "Zed"})

    assert len(df) == 1
    assert df.iloc[0]["class_id"] == 0
    assert "Skipping malformed" in caplog.text or "Skipping out-of-range" in caplog.text


def test_parse_dataset_aggregates_multiple_files(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    images_dir = tmp_path / "images"
    names_file = tmp_path / "mid.names"

    labels_dir.mkdir()
    images_dir.mkdir()

    names_file.write_text("Ahri\nLux\n", encoding="utf-8")

    (labels_dir / "img1.txt").write_text("0 0.50 0.50 0.20 0.20\n", encoding="utf-8")
    (labels_dir / "img2.txt").write_text("1 0.25 0.75 0.10 0.10\n", encoding="utf-8")

    _make_image(images_dir / "img1.png", size=(100, 100))
    _make_image(images_dir / "img2.png", size=(200, 100))

    df = parse_dataset(labels_dir=labels_dir, images_dir=images_dir, names_path=names_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "bbox_area_norm" in df.columns
    assert df["image_id"].nunique() == 2
    assert set(df["class_name"]) == {"Ahri", "Lux"}

    areas = df.sort_values("image_id")["bbox_area_norm"].tolist()
    assert areas[0] == pytest.approx(0.04)
    assert areas[1] == pytest.approx(0.01)