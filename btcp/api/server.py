"""FastAPI server exposing prediction endpoints."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "lstm_model.h5"


def is_model_available(path: Path | None = None) -> bool:
    """Return whether the trained model artifact is a regular file."""
    artifact_path = path or MODEL_PATH
    return artifact_path.is_file()


@app.get("/")
def root():
    return {"status": "ok", "message": "BTC Predictor API is running"}


@app.get("/health")
def health():
    """Return API liveness without requiring model or data dependencies."""
    return {"status": "ok"}


@app.get("/model/status")
def model_status():
    """Return whether the prediction model artifact is available."""
    if not is_model_available():
        return {"model_loaded": False, "reason": "model artifact not found"}

    return {"model_loaded": True, "artifact_path": str(MODEL_PATH)}


@app.get("/predict")
def predict_price():
    """
    1. 실시간 시세를 수집
    2. 전처리 (정규화)
    3. 모델 예측
    4. 결과 반환
    """
    if not is_model_available():
        raise HTTPException(status_code=503, detail="model_not_ready")

    from btcp.data import collector
    from btcp.model import inference
    from btcp.utils import preprocessor

    model = inference.load_trained_model(MODEL_PATH)

    # 1. 실시간 시세 수집
    price_data = collector.get_recent_prices()
    prices = price_data["prices"]

    # 2. 전처리 (여기선 단순 정규화)
    normed_input, mean, std = preprocessor.normalize(prices)

    # 3. 모델 예측
    predicted = inference.predict(model=model, input_data=normed_input)

    # 4. 결과 반환
    return {
        "timestamp": str(price_data["timestamp"]),
        "input_prices": prices,
        "normalized": normed_input.tolist(),
        "predicted": predicted.tolist() if hasattr(predicted, "tolist") else predicted,
        "denormalized_prediction": (predicted * std + mean).tolist()
        if hasattr(predicted, "tolist")
        else float(predicted * std + mean),
    }
