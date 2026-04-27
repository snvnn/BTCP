"""FastAPI server exposing prediction endpoints."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from btcp.config import settings
from btcp.api.serializers import jsonable
from btcp.trading.paper import PaperTradingEngine

app = FastAPI()

MODEL_PATH = settings.model_path
paper_engine = PaperTradingEngine()


def is_model_available(path: Path | None = None) -> bool:
    """Return whether the trained model artifact is a regular file."""
    artifact_path = path or MODEL_PATH
    return artifact_path.is_file()


def get_recent_prices(symbol: str, interval: str, lookback: int):
    """Fetch recent prices lazily so health endpoints do not require data dependencies."""
    from btcp.data import collector

    return collector.get_recent_prices(symbol=symbol, interval=interval, lookback=lookback)


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


@app.post("/paper/start")
def paper_start(
    initial_cash: float = 10_000.0,
    threshold: float = 0.01,
    trade_fraction: float = 0.5,
    fee_rate: float = 0.001,
):
    """Start a fresh in-memory paper trading session."""
    global paper_engine
    paper_engine = PaperTradingEngine(
        initial_cash=initial_cash,
        threshold=threshold,
        trade_fraction=trade_fraction,
        fee_rate=fee_rate,
    )
    paper_engine.start()
    return paper_engine.status()


@app.post("/paper/stop")
def paper_stop():
    """Stop the in-memory paper trading session."""
    paper_engine.stop()
    return paper_engine.status()


@app.get("/paper/status")
def paper_status(current_price: float | None = None):
    """Return the current paper trading status."""
    return paper_engine.status(current_price=current_price)


@app.get("/paper/trades")
def paper_trades():
    """Return simulated paper trading fills."""
    return {"trades": jsonable(paper_engine.trades)}


@app.post("/paper/tick")
def paper_tick(current_price: float, predicted_price: float):
    """Process one manual paper-trading tick."""
    return jsonable(
        paper_engine.on_tick(current_price=current_price, predicted_price=predicted_price)
    )


@app.get("/predict")
def predict_price():
    """Fetch recent prices and return horizon-based model predictions."""
    if not is_model_available():
        raise HTTPException(status_code=503, detail="model_not_ready")

    from btcp.model import inference

    try:
        artifacts = inference.load_inference_artifacts()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="model_not_ready") from exc

    metadata = artifacts.metadata
    seq_length = int(metadata.get("seq_length", settings.seq_length))
    pred_offsets = metadata.get("pred_offsets", [5, 15, 30, 60])

    symbol = metadata.get("symbol", settings.symbol)
    interval = metadata.get("interval", settings.interval)
    price_data = get_recent_prices(symbol=symbol, interval=interval, lookback=seq_length)
    prices = price_data["prices"]
    predictions = inference.predict_with_scaler(
        model=artifacts.model,
        scaler=artifacts.scaler,
        prices=prices,
        pred_offsets=pred_offsets,
        seq_length=seq_length,
    )

    return {
        "symbol": symbol,
        "interval": interval,
        "timestamp": str(price_data["timestamp"]),
        "input": {
            "seq_length": seq_length,
            "last_price": float(prices[-1]),
        },
        "predictions": predictions,
        "model": {
            "artifact": str(settings.model_dir),
            "trained_at": metadata.get("trained_at"),
        },
    }
