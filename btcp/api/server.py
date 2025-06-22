"""FastAPI server exposing prediction endpoints."""

from fastapi import FastAPI

from btcp.data import collector
from btcp.utils import preprocessor
from btcp.model import inference

app = FastAPI()

# Load model once when the server starts
model = inference.load_trained_model()


@app.get("/")
def root():
    return {"status": "ok", "message": "BTC Predictor API is running"}


@app.get("/predict")
def predict_price():
    """
    1. 실시간 시세를 수집
    2. 전처리 (정규화)
    3. 모델 예측
    4. 결과 반환
    """
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
        "normalized": normed_input.tolist(),  # numpy array → list
        "predicted": float(predicted),
        "denormalized_prediction": float(predicted * std + mean)  # 복원된 예측값
    }
