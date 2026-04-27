from pathlib import Path

from btcp.config import Settings


def test_settings_derives_artifact_paths_from_model_dir(tmp_path):
    settings = Settings(model_dir=tmp_path / "models" / "current")

    assert settings.model_path == tmp_path / "models" / "current" / "model.keras"
    assert settings.scaler_path == tmp_path / "models" / "current" / "scaler.joblib"
    assert settings.metadata_path == tmp_path / "models" / "current" / "metadata.json"


def test_settings_defaults_are_project_relative_paths():
    settings = Settings()

    assert settings.symbol == "BTCUSDT"
    assert settings.interval == "1m"
    assert settings.seq_length == 60
    assert settings.data_dir == Path("data")


def test_settings_reads_environment_when_constructed(monkeypatch):
    monkeypatch.setenv("BTCP_SYMBOL", "ETHUSDT")
    monkeypatch.setenv("BTCP_SEQ_LENGTH", "30")

    settings = Settings()

    assert settings.symbol == "ETHUSDT"
    assert settings.seq_length == 30
