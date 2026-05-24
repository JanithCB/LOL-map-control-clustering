
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gui.data_loader import (
    get_cluster_display_name,
    load_app_data,
    load_cluster_infos,
)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_base_csvs(base_dir: Path) -> None:
    _write_csv(
        base_dir / "outputs" / "reports" / "clustering" / "cluster_sizes.csv",
        pd.DataFrame(
            {
                "cluster_label": [0, 1],
                "n_samples": [100, 50],
                "proportion": [0.6667, 0.3333],
            }
        ),
    )

    _write_csv(
        base_dir / "outputs" / "reports" / "clustering" / "cluster_top_features.csv",
        pd.DataFrame(
            {
                "cluster_label": [0, 0, 1, 1],
                "feature": [
                    "bot_lane_count",
                    "dragon_pit_heat",
                    "mid_lane_count",
                    "grouped_mid",
                ],
                "centroid_value": [0.91, 0.84, 0.88, 0.81],
            }
        ),
    )

    _write_csv(
        base_dir / "outputs" / "reports" / "clustering" / "cluster_labels.csv",
        pd.DataFrame(
            {
                "cluster_id": [0, 1],
                "label": ["Bot-side objective setup", "5-man mid group"],
                "short_description": [
                    "Bot-focused setup around dragon-side zones.",
                    "Grouped mid pressure with reduced side-lane spread.",
                ],
            }
        ),
    )

    _write_csv(
        base_dir / "outputs" / "reports" / "clustering" / "representative_samples.csv",
        pd.DataFrame(
            {
                "cluster_label": [0, 0, 1],
                "image_path": [
                    "outputs/samples/c0_a.png",
                    "outputs/samples/c0_b.png",
                    "outputs/samples/c1_a.png",
                ],
                "match_id": ["m1", "m2", "m3"],
                "frame_id": ["10", "25", "30"],
                "timestamp": ["00:30", "01:10", "02:05"],
            }
        ),
    )


def test_load_cluster_infos_normalizes_cluster_id_schema(tmp_path: Path) -> None:
    _make_base_csvs(tmp_path)

    _write_csv(
        tmp_path / "outputs" / "reports" / "comparison" / "qualitative_comparison_notes.csv",
        pd.DataFrame(
            {
                "cluster_label": [0, 1],
                "note": ["dragon pressure and bot river presence", "grouped mid rotation"],
            }
        ),
    )

    cluster_infos = load_cluster_infos(tmp_path)

    assert set(cluster_infos.keys()) == {0, 1}
    assert cluster_infos[0].cluster_id == 0
    assert cluster_infos[1].cluster_id == 1


def test_load_cluster_infos_reads_cluster_labels(tmp_path: Path) -> None:
    _make_base_csvs(tmp_path)

    _write_csv(
        tmp_path / "outputs" / "reports" / "comparison" / "qualitative_comparison_notes.csv",
        pd.DataFrame({"cluster_label": [0], "note": ["bot-side setup"]}),
    )

    cluster_infos = load_cluster_infos(tmp_path)

    assert cluster_infos[0].label == "Bot-side objective setup"
    assert "dragon-side zones" in cluster_infos[0].description
    assert get_cluster_display_name(cluster_infos[0]) == "Cluster 0 – Bot-side objective setup"


def test_load_cluster_infos_uses_n_samples_and_proportion_fallback(tmp_path: Path) -> None:
    _make_base_csvs(tmp_path)

    _write_csv(
        tmp_path / "outputs" / "reports" / "comparison" / "qualitative_comparison_notes.csv",
        pd.DataFrame({"cluster_label": [0], "note": ["bot-side setup"]}),
    )

    cluster_infos = load_cluster_infos(tmp_path)

    assert cluster_infos[0].size == 100
    assert cluster_infos[1].size == 50
    assert cluster_infos[0].pct == 66.67
    assert cluster_infos[1].pct == 33.33


def test_load_cluster_infos_parses_representative_samples(tmp_path: Path) -> None:
    _make_base_csvs(tmp_path)

    _write_csv(
        tmp_path / "outputs" / "reports" / "comparison" / "qualitative_comparison_notes.csv",
        pd.DataFrame({"cluster_label": [0], "note": ["bot-side setup"]}),
    )

    cluster_infos = load_cluster_infos(tmp_path)
    samples = cluster_infos[0].representative_samples

    assert len(samples) == 2
    assert samples[0].image_path == "outputs/samples/c0_a.png"
    assert samples[0].match_id == "m1"
    assert samples[0].frame_id == "10"
    assert samples[0].timestamp == "00:30"
    assert "image_path=outputs/samples/c0_a.png" in samples[0].raw_summary


def test_load_app_data_handles_global_notes_without_cluster_ids(tmp_path: Path) -> None:
    _make_base_csvs(tmp_path)

    _write_csv(
        tmp_path / "outputs" / "reports" / "comparison" / "qualitative_comparison_notes.csv",
        pd.DataFrame(
            {
                "section": ["overview", "comparison"],
                "text": [
                    "Higher-ranked games show more coordinated macro setups.",
                    "Objective-focused states appear earlier in stronger games.",
                ],
            }
        ),
    )

    app_data = load_app_data(tmp_path)

    assert app_data.cluster_infos[0].notes == ""
    assert "overview: overview" in app_data.global_notes
    assert "Higher-ranked games show more coordinated macro setups." in app_data.global_notes
    assert "Objective-focused states appear earlier in stronger games." in app_data.global_notes