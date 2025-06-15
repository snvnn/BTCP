from fastapi import FastAPI
from predictor import data_collector, preprocessor, model_inference

app = FastAPI()

# 서버가 시작되면서 모델을 메모리에 로딩
model = model_inference.load_model()


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
    price_data = data_collector.get_binance_price()
    close_price = price_data["close"]

    # 2. 전처리 (여기선 단순 정규화)
    normed_input, mean, std = preprocessor.normalize([close_price] * 10)

    # 3. 모델 예측
    predicted = model_inference.predict(model, normed_input)

    # 4. 결과 반환
    return {
        "timestamp": str(price_data["timestamp"]),
        "price": close_price,
        "normalized": normed_input,
        "predicted": predicted,
        "denormalized_prediction": predicted * std + mean  # 복원된 예측값
    }
