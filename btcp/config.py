"""Project configuration for BTCP."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    """Runtime settings with environment-backed defaults."""

    symbol: str = field(default_factory=lambda: _env_str("BTCP_SYMBOL", "BTCUSDT"))
    interval: str = field(default_factory=lambda: _env_str("BTCP_INTERVAL", "1m"))
    seq_length: int = field(default_factory=lambda: _env_int("BTCP_SEQ_LENGTH", 60))
    data_dir: Path = field(default_factory=lambda: _env_path("BTCP_DATA_DIR", "data"))
    model_dir: Path = field(
        default_factory=lambda: _env_path("BTCP_MODEL_DIR", "artifacts/models/current")
    )

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.keras"

    @property
    def legacy_model_path(self) -> Path:
        return self.model_dir / "lstm_model.h5"

    @property
    def scaler_path(self) -> Path:
        return self.model_dir / "scaler.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"


settings = Settings()
