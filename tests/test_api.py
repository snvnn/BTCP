from datetime import datetime

from fastapi.testclient import TestClient

import btcp.api.server as server
from btcp.api.server import app
from btcp.model import inference


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_status_reports_missing_model_without_crashing(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_model.h5"
    monkeypatch.setattr(server, "MODEL_PATH", missing_model)

    response = client.get("/model/status")

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert body["reason"] == "model artifact not found"


def test_is_model_available_requires_regular_file(tmp_path):
    model_dir = tmp_path / "lstm_model.h5"
    model_dir.mkdir()

    assert server.is_model_available(model_dir) is False


def test_predict_returns_503_when_model_missing(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing_model.h5"
    monkeypatch.setattr(server, "MODEL_PATH", missing_model)

    response = client.get("/predict")

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_ready"}


def test_predict_uses_saved_scaler_artifacts(monkeypatch, tmp_path):
    model_path = tmp_path / "model.keras"
    model_path.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(server, "MODEL_PATH", model_path)
    captured = {}

    def fake_get_recent_prices(symbol, interval, lookback):
        captured.update({"symbol": symbol, "interval": interval, "lookback": lookback})
        return {
            "timestamp": datetime(2026, 1, 1, 0, 0, 0),
            "prices": [100.0, 101.0, 102.0],
        }

    monkeypatch.setattr(server, "get_recent_prices", fake_get_recent_prices)
    monkeypatch.setattr(
        inference,
        "load_inference_artifacts",
        lambda: inference.InferenceArtifacts(
            model="model",
            scaler="scaler",
            metadata={
                "symbol": "ETHUSDT",
                "interval": "5m",
                "seq_length": 3,
                "pred_offsets": [5, 15],
            },
        ),
    )
    monkeypatch.setattr(
        inference,
        "predict_with_scaler",
        lambda model, scaler, prices, pred_offsets, seq_length: {"5m": 110.0, "15m": 120.0},
    )

    response = client.get("/predict")

    assert response.status_code == 200
    body = response.json()
    assert captured == {"symbol": "ETHUSDT", "interval": "5m", "lookback": 3}
    assert body["input"]["last_price"] == 102.0
    assert body["input"]["seq_length"] == 3
    assert body["predictions"] == {"5m": 110.0, "15m": 120.0}


def test_predict_returns_503_when_artifact_bundle_is_incomplete(monkeypatch, tmp_path):
    model_path = tmp_path / "model.keras"
    model_path.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(server, "MODEL_PATH", model_path)
    monkeypatch.setattr(
        inference,
        "load_inference_artifacts",
        lambda: (_ for _ in ()).throw(FileNotFoundError("scaler.joblib")),
    )

    response = client.get("/predict")

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_ready"}
