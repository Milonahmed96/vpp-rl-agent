"""Tests for the training module's data helpers (excludes the heavy PPO run)."""

from __future__ import annotations

import numpy as np

from src.agent.train import build_feature_frame, load_merged_data
from src.features.engineer import AUX_COLUMNS, FEATURE_NAMES
from src.ingestion.synthetic import generate_synthetic_merged


def test_load_merged_data_reads_existing_csv(tmp_path):
    path = tmp_path / "merged.csv"
    generate_synthetic_merged(n_days=5).to_csv(path, index=False)
    frame = load_merged_data(str(path))
    assert len(frame) == 5 * 48


def test_load_merged_data_generates_synthetic_when_missing(tmp_path):
    path = tmp_path / "absent.csv"
    frame = load_merged_data(str(path))
    assert not frame.empty
    # The fallback also persists the generated data for reuse.
    assert path.exists()


def test_build_feature_frame_has_features_aux_and_no_nan(tmp_path, monkeypatch):
    # Keep the fitted scaler inside tmp to avoid touching the repo's data dir.
    monkeypatch.setattr(
        "src.agent.train.FeatureEngineer",
        lambda: __import__(
            "src.features.engineer", fromlist=["FeatureEngineer"]
        ).FeatureEngineer(scaler_path=str(tmp_path / "scaler.pkl")),
    )
    merged = generate_synthetic_merged(n_days=10)
    features = build_feature_frame(merged)
    for col in (*FEATURE_NAMES, *AUX_COLUMNS):
        assert col in features.columns
    assert not np.isnan(features[list(FEATURE_NAMES)].to_numpy()).any()
