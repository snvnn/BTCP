from fastapi.testclient import TestClient

import btcp.api.server as server
from btcp.api.server import app


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
