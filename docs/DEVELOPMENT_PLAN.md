# BTCP 절차별 개발 계획

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task.

**Goal:** BTCP를 현재 프로토타입 상태에서 실행 가능하고 검증 가능한 BTC 예측 API/대시보드/모의투자 데모로 안정화한다.

**Architecture:** 기존 `btcp/`, `dashboard/`, `docker/` 구조를 유지하면서 설정, artifact, 전처리, API, 평가, 대시보드, 테스트를 단계적으로 정리한다. 전면 개편은 하지 않고, 먼저 작은 변경으로 실행 가능성과 평가 가능성을 확보한다.

**Tech Stack:** Python, FastAPI, Uvicorn, Streamlit, Pandas, NumPy, scikit-learn, TensorFlow/Keras, pytest, Docker.

---

## 개발 원칙

- 한 작업은 작게 유지한다.
- 각 단계는 검증 명령을 포함한다.
- 기존 동작을 망가뜨리지 않는 방향으로 진행한다.
- 모델 성능 개선보다 먼저 학습/추론 일관성과 평가 신뢰성을 확보한다.
- 실제 매매 기능은 만들지 않는다. 모의투자는 실제 주문 없이 가상 잔고와 가상 체결만 다룬다.
- 각 단계 완료 후 커밋한다.

## Phase 0: 문서화 및 저장소 정리

### Task 0.1: `.idea/` 추적 제거 및 `.gitignore` 정리

**Objective:** IDE 개인 설정 파일이 저장소에 포함되지 않도록 정리한다.

**Files:**

- Modify: `.gitignore`
- Remove from Git index: `.idea/*`

**Steps:**

1. `.gitignore`에 다음 항목이 있는지 확인한다.

```gitignore
.idea/
.venv/
venv/
__pycache__/
*.pyc
*.h5
*.keras
*.joblib
artifacts/
data/*.csv
```

2. Git index에서 `.idea/`를 제거한다.

```bash
git rm -r --cached .idea
```

3. 검증한다.

```bash
git status --short
```

Expected:

- `.idea/` 파일들이 삭제 예정으로 보인다.
- 로컬 디렉터리 자체는 남아 있어도 된다.

4. 커밋한다.

```bash
git add .gitignore
git commit -m "chore: stop tracking IDE files"
```

### Task 0.2: README/설계서/개발계획 문서 검토

**Objective:** 새로 작성한 문서가 현재 코드와 모순되지 않는지 확인한다.

**Files:**

- Review: `README.md`
- Review: `docs/DESIGN.md`
- Review: `docs/DEVELOPMENT_PLAN.md`

**Steps:**

1. Markdown 파일 목록을 확인한다.

```bash
find . -maxdepth 3 -name '*.md' -print
```

2. 링크가 맞는지 확인한다.

```bash
grep -n "docs/" README.md
```

3. 커밋한다.

```bash
git add README.md docs/DESIGN.md docs/DEVELOPMENT_PLAN.md
git commit -m "docs: add project overview and development plan"
```

## Phase 1: 설정과 실행 가능성 확보

### Task 1.1: 공통 설정 모듈 추가

**Objective:** 하드코딩된 경로와 기본값을 한 곳에서 관리한다.

**Files:**

- Create: `btcp/config.py`
- Test: `tests/test_config.py`

**Implementation outline:**

`btcp/config.py`에 다음 설정을 추가한다.

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    symbol: str = os.getenv("BTCP_SYMBOL", "BTCUSDT")
    interval: str = os.getenv("BTCP_INTERVAL", "1m")
    seq_length: int = int(os.getenv("BTCP_SEQ_LENGTH", "60"))
    data_dir: Path = Path(os.getenv("BTCP_DATA_DIR", "data"))
    model_dir: Path = Path(os.getenv("BTCP_MODEL_DIR", "artifacts/models/current"))

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.keras"

    @property
    def scaler_path(self) -> Path:
        return self.model_dir / "scaler.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"


settings = Settings()
```

**Verification:**

```bash
python - <<'PY'
from btcp.config import settings
print(settings.symbol)
print(settings.model_path)
PY
```

Expected:

```text
BTCUSDT
artifacts/models/current/model.keras
```

**Commit:**

```bash
git add btcp/config.py tests/test_config.py
git commit -m "feat: add project settings"
```

### Task 1.2: API health endpoint 추가

**Objective:** 모델 파일이 없어도 서버 생존 여부를 확인할 수 있게 한다.

**Files:**

- Modify: `btcp/api/server.py`
- Test: `tests/test_api.py`

**Required behavior:**

- `GET /health` returns `{"status": "ok"}`.
- API module import should not fail just because model file is missing.

**Implementation notes:**

현재 `server.py`는 module import 시점에 모델을 로드한다.

```python
model = inference.load_trained_model()
```

이 동작을 제거하고 lazy loading 또는 app state 기반으로 바꾼다.

**Verification:**

```bash
uvicorn btcp.api.server:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

**Commit:**

```bash
git add btcp/api/server.py tests/test_api.py
git commit -m "feat: add API health endpoint"
```

### Task 1.3: 모델 상태 endpoint 추가

**Objective:** 현재 모델 artifact가 준비되었는지 API로 확인할 수 있게 한다.

**Files:**

- Modify: `btcp/api/server.py`
- Modify/Create: `btcp/model/inference.py`
- Test: `tests/test_api.py`

**Required behavior:**

- `GET /model/status` returns `model_loaded: false` when artifact is missing.
- 서버는 모델이 없어도 실행된다.

**Response example:**

```json
{
  "model_loaded": false,
  "reason": "model artifact not found"
}
```

**Verification:**

```bash
curl http://127.0.0.1:8000/model/status
```

Expected:

- HTTP 200
- `model_loaded` field exists

**Commit:**

```bash
git add btcp/api/server.py btcp/model/inference.py tests/test_api.py
git commit -m "feat: expose model status endpoint"
```

## Phase 2: 데이터 수집 안정화

### Task 2.1: Binance kline client 통합

**Objective:** `collector.py`, `crawler.py`, `realtime.py`에 흩어진 Binance 요청 로직을 통합한다.

**Files:**

- Create: `btcp/data/binance_client.py`
- Modify: `btcp/data/collector.py`
- Modify: `btcp/data/crawler.py`
- Modify: `btcp/data/realtime.py`
- Test: `tests/test_binance_client.py`

**Required behavior:**

- 모든 HTTP 요청에 timeout을 둔다.
- HTTP error를 명확히 raise한다.
- kline response를 DataFrame으로 바꾸는 함수를 공통화한다.

**Verification:**

```bash
python -m py_compile btcp/data/binance_client.py btcp/data/collector.py btcp/data/crawler.py btcp/data/realtime.py
pytest tests/test_binance_client.py -v
```

**Commit:**

```bash
git add btcp/data tests/test_binance_client.py
git commit -m "refactor: centralize Binance kline client"
```

### Task 2.2: 데이터 저장 경로 설정화

**Objective:** CSV가 실행 위치에 따라 임의 위치에 저장되지 않도록 한다.

**Files:**

- Modify: `btcp/data/collector.py`
- Modify: `btcp/data/crawler.py`
- Modify: `btcp/config.py`

**Required behavior:**

- 기본 저장 경로는 `data/`다.
- 저장 전 디렉터리를 생성한다.
- 파일명은 symbol/interval을 포함한다.

**Verification:**

```bash
python - <<'PY'
from btcp.config import settings
settings.data_dir.mkdir(exist_ok=True)
print(settings.data_dir)
PY
```

**Commit:**

```bash
git add btcp/config.py btcp/data/collector.py btcp/data/crawler.py
git commit -m "fix: configure data output paths"
```

## Phase 3: 학습/추론 artifact 통일

### Task 3.1: trainer 경로 하드코딩 제거

**Objective:** `/home/yunh/BTCP/...` 같은 로컬 절대 경로를 제거한다.

**Files:**

- Modify: `btcp/model/trainer.py`
- Modify: `btcp/config.py`

**Required behavior:**

- 데이터 경로는 CLI 인자 또는 settings에서 받는다.
- 모델 저장 경로는 `settings.model_dir`를 사용한다.

**Verification:**

```bash
grep -R "/home/yunh" -n btcp || true
```

Expected:

- No results.

**Commit:**

```bash
git add btcp/model/trainer.py btcp/config.py
git commit -m "fix: remove hardcoded training paths"
```

### Task 3.2: scaler artifact 저장

**Objective:** 학습 시 사용한 scaler를 추론에서도 동일하게 사용하도록 저장한다.

**Files:**

- Modify: `btcp/model/trainer.py`
- Modify: `requirements.txt`
- Test: `tests/test_training_artifacts.py`

**Required behavior:**

- 학습 완료 후 `model.keras`, `scaler.joblib`, `metadata.json` 저장
- `metadata.json`에는 seq_length, pred_offsets, symbol, interval 포함

**Implementation note:**

`joblib`이 필요하면 `requirements.txt`에 추가한다.

**Verification:**

작은 샘플 데이터로 artifact 저장 테스트를 한다.

```bash
pytest tests/test_training_artifacts.py -v
```

**Commit:**

```bash
git add btcp/model/trainer.py requirements.txt tests/test_training_artifacts.py
git commit -m "feat: save model scaler and metadata artifacts"
```

### Task 3.3: inference에서 model+scaler 함께 로드

**Objective:** 학습과 동일한 전처리를 추론에 적용한다.

**Files:**

- Modify: `btcp/model/inference.py`
- Test: `tests/test_inference.py`

**Required behavior:**

- artifact 디렉터리에서 model/scaler/metadata를 함께 로드한다.
- 입력 길이가 metadata의 seq_length와 다르면 명확한 error를 낸다.
- 예측 결과는 horizon별 dict로 반환한다.

**Expected output:**

```python
{
    "5m": 65000.0,
    "15m": 65100.0,
    "30m": 65200.0,
    "60m": 65300.0,
}
```

**Verification:**

```bash
pytest tests/test_inference.py -v
```

**Commit:**

```bash
git add btcp/model/inference.py tests/test_inference.py
git commit -m "feat: load complete inference artifacts"
```

## Phase 4: API 예측 스키마 정리

### Task 4.1: `/predict` 모델 없음 상태 처리

**Objective:** 모델이 없을 때 500 대신 503을 반환한다.

**Files:**

- Modify: `btcp/api/server.py`
- Test: `tests/test_api.py`

**Required behavior:**

- 모델 artifact가 없으면 `GET /predict` returns HTTP 503.
- body contains `detail: model_not_ready`.

**Verification:**

```bash
pytest tests/test_api.py::test_predict_returns_503_when_model_missing -v
```

**Commit:**

```bash
git add btcp/api/server.py tests/test_api.py
git commit -m "fix: return 503 when prediction model is missing"
```

### Task 4.2: `/predict` horizon별 응답 구현

**Objective:** 예측 결과를 5m/15m/30m/60m로 명확히 반환한다.

**Files:**

- Modify: `btcp/api/server.py`
- Modify: `btcp/model/inference.py`
- Test: `tests/test_api.py`

**Required response shape:**

```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "timestamp": "...",
  "input": {
    "seq_length": 60,
    "last_price": 65000.0
  },
  "predictions": {
    "5m": 65020.0,
    "15m": 65100.0,
    "30m": 65200.0,
    "60m": 65300.0
  }
}
```

**Verification:**

```bash
pytest tests/test_api.py -v
```

**Commit:**

```bash
git add btcp/api/server.py btcp/model/inference.py tests/test_api.py
git commit -m "feat: return horizon-based predictions"
```

## Phase 5: baseline 및 평가 도입

### Task 5.1: baseline 모델 추가

**Objective:** LSTM 성능을 비교할 기준 모델을 만든다.

**Files:**

- Create: `btcp/model/baseline.py`
- Test: `tests/test_baseline.py`

**Required behavior:**

- `last_price_baseline(prices, offsets)` returns current price for all offsets.
- `moving_average_baseline(prices, offsets, window)` returns recent moving average for all offsets.

**Verification:**

```bash
pytest tests/test_baseline.py -v
```

**Commit:**

```bash
git add btcp/model/baseline.py tests/test_baseline.py
git commit -m "feat: add baseline prediction models"
```

### Task 5.2: 평가 지표 추가

**Objective:** horizon별 MAE/RMSE/MAPE/directional accuracy를 계산한다.

**Files:**

- Modify/Create: `btcp/model/metrics.py`
- Modify: `btcp/model/evaluator.py`
- Test: `tests/test_metrics.py`

**Required behavior:**

- numpy array/list 입력 지원
- horizon별 metric dict 반환
- 0 나눗셈 방어

**Verification:**

```bash
pytest tests/test_metrics.py -v
```

**Commit:**

```bash
git add btcp/model/metrics.py btcp/model/evaluator.py tests/test_metrics.py
git commit -m "feat: add prediction evaluation metrics"
```

### Task 5.3: 최근 데이터 평가 리포트 생성

**Objective:** LSTM과 baseline을 같은 데이터에서 비교한다.

**Files:**

- Modify: `btcp/model/evaluator.py`
- Create: `artifacts/reports/.gitkeep` or keep ignored

**Required behavior:**

- `python -m btcp.model.evaluator --recent` 실행 시 metric summary 출력
- 가능하면 `artifacts/reports/latest_metrics.json` 저장

**Verification:**

```bash
python -m btcp.model.evaluator --recent
```

Expected:

- horizon별 metric 출력
- baseline 비교 포함

**Commit:**

```bash
git add btcp/model/evaluator.py
git commit -m "feat: compare model predictions against baselines"
```


## Phase 6: 백테스트 및 모의투자 핵심 도메인 추가

### Task 6.1: VirtualPortfolio 모델 추가

**Objective:** 실제 주문 없이 가상 현금과 BTC 포지션을 관리한다.

**Files:**

- Create: `btcp/trading/__init__.py`
- Create: `btcp/trading/portfolio.py`
- Test: `tests/test_portfolio.py`

**Required behavior:**

- 초기 현금 잔고를 설정한다.
- 시장가 매수 시 현금이 줄고 BTC 수량이 늘어난다.
- 시장가 매도 시 BTC 수량이 줄고 현금이 늘어난다.
- 수수료를 반영한다.
- 총 평가금액과 실현/미실현 손익을 계산한다.

**Verification:**

```bash
pytest tests/test_portfolio.py -v
```

**Commit:**

```bash
git add btcp/trading/portfolio.py btcp/trading/__init__.py tests/test_portfolio.py
git commit -m "feat: add virtual trading portfolio"
```

### Task 6.2: Strategy 인터페이스와 단순 예측 전략 추가

**Objective:** 모델 예측값을 buy/sell/hold 신호로 변환한다.

**Files:**

- Create: `btcp/trading/strategy.py`
- Test: `tests/test_strategy.py`

**Required behavior:**

- 현재가와 특정 horizon 예측값을 입력받는다.
- 기대 수익률이 threshold보다 크면 `buy`를 반환한다.
- 기대 수익률이 -threshold보다 작으면 `sell`을 반환한다.
- 그 외에는 `hold`를 반환한다.

**Verification:**

```bash
pytest tests/test_strategy.py -v
```

**Commit:**

```bash
git add btcp/trading/strategy.py tests/test_strategy.py
git commit -m "feat: convert predictions to trading signals"
```

### Task 6.3: Backtester 추가

**Objective:** 과거 데이터에서 전략을 실행해 모의 성과를 계산한다.

**Files:**

- Create: `btcp/trading/backtester.py`
- Test: `tests/test_backtester.py`

**Required behavior:**

- 캔들 데이터와 예측값 또는 전략을 입력받는다.
- 각 시점마다 signal을 계산한다.
- VirtualPortfolio에 가상 주문을 적용한다.
- final equity, total return, max drawdown, trade count를 반환한다.

**Verification:**

```bash
pytest tests/test_backtester.py -v
```

**Commit:**

```bash
git add btcp/trading/backtester.py tests/test_backtester.py
git commit -m "feat: add backtesting engine"
```

### Task 6.4: PaperTradingEngine 초안 추가

**Objective:** 실시간 가격/예측값을 사용해 가상 포트폴리오를 업데이트한다.

**Files:**

- Create: `btcp/trading/paper.py`
- Test: `tests/test_paper.py`

**Required behavior:**

- start/stop 상태를 가진다.
- tick 단위로 현재가와 예측값을 받아 signal을 계산한다.
- signal에 따라 가상 포트폴리오를 업데이트한다.
- 거래 내역을 메모리에 기록한다.

**Verification:**

```bash
pytest tests/test_paper.py -v
```

**Commit:**

```bash
git add btcp/trading/paper.py tests/test_paper.py
git commit -m "feat: add paper trading engine"
```

## Phase 7: Streamlit 대시보드 개선

### Task 7.1: API URL 환경변수화

**Objective:** 대시보드가 실행 환경에 따라 API 주소를 바꿀 수 있게 한다.

**Files:**

- Modify: `dashboard/streamlit_app.py`

**Required behavior:**

- `BTCP_API_URL` 환경변수를 읽는다.
- 기본값은 `http://localhost:8000`이다.
- predict endpoint는 `{BTCP_API_URL}/predict`로 구성한다.

**Verification:**

```bash
BTCP_API_URL=http://127.0.0.1:8000 streamlit run dashboard/streamlit_app.py
```

**Commit:**

```bash
git add dashboard/streamlit_app.py
git commit -m "fix: configure dashboard API URL"
```

### Task 7.2: horizon별 예측 UI 추가

**Objective:** 5m/15m/30m/60m 예측값을 명확히 표시한다.

**Files:**

- Modify: `dashboard/streamlit_app.py`

**Required behavior:**

- `predictions` dict를 표로 표시
- 최근 가격과 horizon별 예측값을 차트로 표시
- 모델 없음/API 오류를 사용자 친화적으로 표시

**Verification:**

```bash
streamlit run dashboard/streamlit_app.py
```

Manual check:

- API 오류 시 에러 박스 표시
- 정상 응답 시 예측 테이블 표시

**Commit:**

```bash
git add dashboard/streamlit_app.py
git commit -m "feat: show horizon predictions in dashboard"
```

## Phase 8: Docker 및 실행 절차 정리

### Task 8.1: docker-compose 루트 기준 실행 정리

**Objective:** compose 실행 위치와 volume 경로 혼란을 줄인다.

**Files:**

- Move/Modify: `docker/docker-compose.yml` or create root `docker-compose.yml`
- Modify: `README.md`

**Recommended direction:**

루트에 `docker-compose.yml`을 두는 것을 권장한다.

**Required behavior:**

```bash
docker-compose up --build
```

을 저장소 루트에서 실행한다.

**Verification:**

```bash
docker-compose config
```

Expected:

- config validation succeeds.

**Commit:**

```bash
git add docker-compose.yml docker/ README.md
git commit -m "chore: simplify docker compose layout"
```

### Task 8.2: Dockerfile 개발/운영 모드 정리

**Objective:** API container가 reload와 artifact volume을 명확히 다루도록 한다.

**Files:**

- Modify: `docker/Dockerfile.api`
- Modify: `docker/Dockerfile.streamlit`
- Modify: `docker-compose.yml`

**Required behavior:**

- `/app/artifacts` volume mount
- `/app/data` volume mount
- API server command 명확화

**Verification:**

```bash
docker-compose build
```

**Commit:**

```bash
git add docker/Dockerfile.api docker/Dockerfile.streamlit docker-compose.yml
git commit -m "chore: align docker paths with artifacts and data"
```

## Phase 9: 테스트와 CI

### Task 9.1: pytest 개발 의존성 추가

**Objective:** 테스트 실행 환경을 만든다.

**Files:**

- Create: `requirements-dev.txt` or modify `requirements.txt`
- Create: `tests/`

**Required behavior:**

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

**Commit:**

```bash
git add requirements-dev.txt tests
git commit -m "test: add pytest setup"
```

### Task 9.2: GitHub Actions CI 추가

**Objective:** push/PR마다 기본 검증을 자동화한다.

**Files:**

- Create: `.github/workflows/ci.yml`

**Required checks:**

- checkout
- setup-python 3.11
- install dependencies
- `python -m py_compile` for project files
- `pytest`

**Verification:**

```bash
python -m py_compile $(git ls-files '*.py')
pytest
```

**Commit:**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add Python test workflow"
```

## Phase 10: 1차 완료 기준

1차 안정화 완료 조건:

- `README.md`만 보고 로컬 실행 절차를 이해할 수 있다.
- `GET /health`는 모델 파일 없이도 성공한다.
- `GET /model/status`로 모델 준비 여부를 알 수 있다.
- 학습 artifact가 `model.keras`, `scaler.joblib`, `metadata.json`으로 저장된다.
- 추론은 학습 때 저장한 scaler를 사용한다.
- `/predict`는 horizon별 응답을 반환한다.
- 모델 없음 상태에서 `/predict`는 503을 반환한다.
- baseline 대비 평가 결과를 확인할 수 있다.
- Streamlit에서 horizon별 예측값을 볼 수 있다.
- 가상 포트폴리오와 백테스트 엔진의 기본 테스트가 통과한다.
- 모의투자 상태/결과를 API 또는 대시보드로 확장할 수 있는 도메인 모델이 준비된다.
- 최소 pytest가 통과한다.
- CI가 기본 검증을 수행한다.

## Phase 11: 이후 의사결정

1차 안정화 후 다음 질문에 답한다.

1. LSTM이 naive baseline보다 나은가?
2. 가격 예측보다 방향성 예측이 더 유의미한가?
3. 대시보드가 실제로 유용한가?
4. 백테스트를 추가할 가치가 있는가?
5. 이 프로젝트를 포트폴리오 데모로 마감할 것인가, 장기 플랫폼으로 키울 것인가?

답변에 따라 다음 중 하나를 선택한다.

- **데모 완성:** README, 스크린샷, Docker 실행 안정화 중심으로 마감
- **분석 도구 확장:** baseline, 지표, 백테스트, 리포트 강화
- **플랫폼 전환:** `apps/`, `packages/`, `pipelines/`, `artifacts/` 구조로 전면 개편
- **자동매매 기반 확장:** 단, 충분한 백테스트와 리스크 관리 설계 후 진행
