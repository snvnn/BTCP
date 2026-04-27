import pickle

import numpy as np
import pytest

from btcp.model import inference


class FakeScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float) / 100.0

    def inverse_transform(self, values):
        return np.asarray(values, dtype=float) * 100.0


class FakeModel:
    def predict(self, values):
        assert values.shape == (1, 3, 1)
        return np.array([[1.1, 1.2]])


class ShortOutputModel:
    def predict(self, values):
        return np.array([[1.1]])


def test_inference_module_imports_without_tensorflow():
    assert inference.SEQ_LENGTH == 60


def test_predict_with_scaler_returns_denormalized_values_by_horizon():
    result = inference.predict_with_scaler(
        model=FakeModel(),
        scaler=FakeScaler(),
        prices=[100.0, 101.0, 102.0],
        pred_offsets=[5, 15],
        seq_length=3,
    )

    assert result == {"5m": pytest.approx(110.0), "15m": pytest.approx(120.0)}


def test_predict_with_scaler_rejects_wrong_sequence_length():
    with pytest.raises(ValueError, match="Expected 3 prices"):
        inference.predict_with_scaler(
            model=FakeModel(),
            scaler=FakeScaler(),
            prices=[100.0, 101.0],
            pred_offsets=[5, 15],
            seq_length=3,
        )


def test_predict_with_scaler_validates_prediction_horizon_count():
    with pytest.raises(ValueError, match="Expected 2 predictions, got 1"):
        inference.predict_with_scaler(
            model=ShortOutputModel(),
            scaler=FakeScaler(),
            prices=[100.0, 101.0, 102.0],
            pred_offsets=[5, 15],
            seq_length=3,
        )


def test_load_inference_artifacts_loads_model_scaler_and_metadata(tmp_path):
    artifact_dir = tmp_path / "current"
    artifact_dir.mkdir()
    model_path = artifact_dir / "model.keras"
    model_path.write_text("fake model", encoding="utf-8")
    scaler = FakeScaler()
    with (artifact_dir / "scaler.joblib").open("wb") as f:
        pickle.dump(scaler, f)
    (artifact_dir / "metadata.json").write_text(
        '{"seq_length": 3, "pred_offsets": [5, 15]}', encoding="utf-8"
    )

    artifacts = inference.load_inference_artifacts(
        artifact_dir=artifact_dir,
        model_loader=lambda path: ("model", path),
    )

    assert artifacts.model == ("model", model_path)
    assert isinstance(artifacts.scaler, FakeScaler)
    assert artifacts.metadata == {"seq_length": 3, "pred_offsets": [5, 15]}
