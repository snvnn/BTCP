# BTCP 설계서

## 1. 문서 목적

이 문서는 BTCP 프로젝트의 현재 목표, 시스템 구성, 책임 분리, 데이터 흐름, 모델 학습/추론 구조, API/대시보드 설계 방향을 정리한다.

현재 프로젝트는 초기 프로토타입이므로, 본 설계서는 완성된 최종 아키텍처가 아니라 **기존 코드를 개선하여 신뢰 가능한 예측 데모로 발전시키기 위한 기준 설계**다.

## 2. 프로젝트 목표

BTCP의 1차 목표는 다음과 같다.

> Binance BTC/USDT 1분봉 데이터를 기반으로 단기 가격 예측 모델을 학습하고, 예측 결과를 API와 대시보드로 확인하며, 모의투자를 통해 예측의 의사결정 가치를 검증할 수 있는 작동 가능한 프로토타입을 만든다.

1차 목표에서 중요한 것은 “복잡한 모델”이 아니라 다음이다.

- 재현 가능한 데이터 수집
- 일관된 전처리
- 학습/추론 artifact 관리
- baseline 대비 평가
- API와 대시보드의 안정적 실행

## 3. 비목표

현재 단계에서 바로 하지 않을 것:

- 실제 자동매매 실행
- 실거래 주문 API 연동
- 다종목 포트폴리오 최적화
- 대규모 MLOps 플랫폼 구축
- 복잡한 Transformer 모델 도입
- 사용자 계정/인증 시스템 구축

이 기능들은 모델 가치와 제품 방향이 검증된 뒤 검토한다.

## 4. 사용자 시나리오

### 4.1 개발자/운영자

1. Binance에서 BTC/USDT 데이터를 수집한다.
2. 학습 스크립트를 실행한다.
3. 모델과 scaler artifact가 저장된다.
4. 평가 스크립트로 baseline 대비 성능을 확인한다.
5. API 서버를 실행한다.
6. Streamlit 대시보드에서 예측 결과를 확인한다.

### 4.2 대시보드 사용자

1. 웹 대시보드에 접속한다.
2. 현재 BTC/USDT 최근 가격 흐름을 확인한다.
3. 5분/15분/30분/60분 후 예측값을 확인한다.
4. 모델의 최근 평가 지표를 확인한다.
5. 예측값을 투자 조언이 아닌 참고용 정보로만 사용한다.

## 4.3 모의투자 사용자

1. 초기 가상 잔고를 설정한다.
2. 모델 예측 또는 간단한 규칙 기반 전략을 선택한다.
3. 과거 데이터로 백테스트를 실행한다.
4. 수익률, 최대 낙폭, 거래 횟수, 승률, 수수료 영향을 확인한다.
5. 실시간 모의투자를 시작한다.
6. 대시보드에서 가상 잔고, 보유 포지션, 실현/미실현 손익, 거래 내역을 확인한다.

## 5. 시스템 개요

```text
Binance API
    |
    v
Data Collector
    |
    v
Raw/Processed Data
    |
    +--> Trainer -----> Model Artifact
    |                    Scaler Artifact
    |                    Metadata
    |
    +--> Evaluator ----> Metrics Report
                         Baseline Comparison

Model Artifact + Recent Prices
    |
    v
FastAPI Prediction API
    |
    +--> Streamlit Dashboard
    |
    v
Trading Layer
    |
    +--> Backtester
    +--> Paper Trading Engine
    +--> Virtual Portfolio
```

## 6. 주요 모듈 책임

### 6.1 `btcp.data`

책임:

- Binance kline 데이터 수집
- 최근 가격 조회
- historical 데이터 저장
- 수집 실패/타임아웃/레이트리밋 처리

현재 파일:

- `btcp/data/collector.py`
- `btcp/data/crawler.py`
- `btcp/data/realtime.py`

개선 방향:

- 중복된 Binance 호출 로직 통합
- timeout 명시
- response validation 추가
- CSV 저장 경로 설정화
- 장기적으로는 DB 또는 Parquet 저장 고려

### 6.2 `btcp.utils`

책임:

- 전처리 공통 유틸 제공
- 학습/추론에서 동일한 변환 규칙 사용

현재 파일:

- `btcp/utils/preprocessor.py`

개선 방향:

- 현재 z-score normalize와 학습용 MinMaxScaler 불일치 제거
- scaler 저장/로드를 모델 artifact와 함께 관리
- 입력 sequence 길이 검증 추가

### 6.3 `btcp.model`

책임:

- 모델 학습
- 모델 artifact 저장
- 모델 로드
- 예측 실행
- 평가 및 baseline 비교

현재 파일:

- `btcp/model/trainer.py`
- `btcp/model/inference.py`
- `btcp/model/evaluator.py`

개선 방향:

- hardcoded path 제거
- artifact 구조 통일
- 다중 horizon 예측 명시
- baseline 모델 추가
- 평가 리포트 생성

### 6.4 `btcp.api`

책임:

- 모델 상태 확인
- 최근 데이터 수집 또는 조회
- 예측 API 제공
- 실패 상태를 명확한 HTTP 응답으로 반환

현재 파일:

- `btcp/api/server.py`

개선 방향:

- `/health` 추가
- `/model/status` 추가
- `/predict` 응답 스키마 정리
- 모델 파일 없음 상태에서 서버 시작 가능하도록 변경
- API 파라미터 검증 추가

### 6.5 `btcp.trading`

책임:

- 가상 잔고와 포지션 관리
- 주문/체결/거래 내역 모델링
- 예측값을 매매 신호로 변환
- 과거 데이터 기반 백테스트 실행
- 실시간 모의투자 상태 업데이트

권장 신규 파일:

- `btcp/trading/portfolio.py`
- `btcp/trading/order.py`
- `btcp/trading/strategy.py`
- `btcp/trading/backtester.py`
- `btcp/trading/paper.py`

초기 범위:

- 현물 long-only 전략만 지원
- 시장가 가정
- 수수료율 설정 가능
- 슬리피지는 고정 bps로 단순화
- 실거래 주문 API 연동 없음

### 6.6 `dashboard`

책임:

- API에서 예측 결과 조회
- 최근 가격과 예측값 시각화
- 평가 지표 표시

현재 파일:

- `dashboard/streamlit_app.py`

개선 방향:

- API URL 환경변수화
- horizon별 예측값 표시
- 오류 메시지 개선
- 평가 지표 영역 추가

## 7. 데이터 설계

### 7.1 입력 데이터

Binance kline API 기준 주요 필드:

| 필드 | 설명 |
|---|---|
| timestamp | 캔들 시작 시간 |
| open | 시가 |
| high | 고가 |
| low | 저가 |
| close | 종가 |
| volume | 거래량 |
| close_time | 캔들 종료 시간 |
| number_of_trades | 거래 수 |

현재 모델은 `close` 가격만 사용한다.

### 7.2 학습 데이터

기본 입력:

- 최근 60개 1분봉 종가

기본 출력:

- 5분 후 가격
- 15분 후 가격
- 30분 후 가격
- 60분 후 가격

설정값:

```text
SEQ_LENGTH = 60
PRED_OFFSETS = [5, 15, 30, 60]
```

### 7.3 artifact 구조

권장 구조:

```text
artifacts/
  models/
    current/
      model.keras
      scaler.joblib
      metadata.json
    runs/
      20260426-1830/
        model.keras
        scaler.joblib
        metadata.json
        metrics.json
```

`metadata.json` 예시:

```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "seq_length": 60,
  "pred_offsets": [5, 15, 30, 60],
  "feature_columns": ["close"],
  "target": "close",
  "scaler": "MinMaxScaler",
  "trained_at": "2026-04-26T18:30:00Z",
  "data_start": "2024-01-01T00:00:00Z",
  "data_end": "2026-04-26T00:00:00Z"
}
```

## 8. 모델 설계

### 8.1 1차 모델

현재 LSTM 모델을 유지한다.

입력 shape:

```text
(batch_size, 60, 1)
```

출력 shape:

```text
(batch_size, 4)
```

출력 순서:

```text
[5m, 15m, 30m, 60m]
```

### 8.2 baseline 모델

모델 성능을 판단하기 위해 baseline을 반드시 추가한다.

필수 baseline:

1. Last Price Baseline
   - 모든 horizon에 현재가를 그대로 예측값으로 사용
2. Moving Average Baseline
   - 최근 N개 평균을 예측값으로 사용

LSTM이 baseline보다 낫지 않다면 모델 구조보다 feature/target/evaluation 설계를 먼저 재검토한다.

## 9. 평가 설계

### 9.1 회귀 지표

horizon별로 계산한다.

- MAE
- RMSE
- MAPE

### 9.2 방향성 지표

가격 자체 예측보다 방향성 예측이 더 유용할 수 있으므로 다음을 계산한다.

- Directional Accuracy
- Up/Down confusion matrix

### 9.3 트레이딩 관점 확장 지표

2차 이후 검토:

- fee/slippage 반영 수익률
- cumulative return
- max drawdown
- Sharpe ratio
- buy-and-hold 대비 성능


## 10. 모의투자 설계

모의투자는 BTCP의 테스트/검증 기능에서 핵심 역할을 한다. 단순 예측 오차가 낮더라도 실제 의사결정에 도움이 되지 않을 수 있으므로, 예측값을 가상 매매로 연결해 성과를 확인한다.

### 10.1 핵심 개념

| 개념 | 설명 |
|---|---|
| VirtualPortfolio | 가상 현금, 보유 BTC, 평균 단가, 실현/미실현 손익 관리 |
| Order | 매수/매도 요청 |
| Fill | 체결 결과. 가격, 수량, 수수료 포함 |
| TradeLog | 모든 가상 거래 기록 |
| Strategy | 예측값과 현재가를 입력받아 buy/sell/hold 결정 |
| Backtester | 과거 캔들 데이터에 전략을 적용해 성과 계산 |
| PaperTradingEngine | 실시간 가격과 예측값을 사용해 가상 포트폴리오 업데이트 |

### 10.2 1차 전략 범위

초기 전략은 복잡하게 만들지 않는다.

예시:

```text
if predicted_15m_return > threshold:
    BUY
elif predicted_15m_return < -threshold:
    SELL
else:
    HOLD
```

초기 제약:

- spot long-only
- leverage 없음
- short 없음
- 시장가 체결 가정
- 수수료율 기본값 0.1%
- 슬리피지 기본값 0.0~0.05% 범위에서 설정

### 10.3 백테스트 결과 지표

- final equity
- total return
- buy-and-hold 대비 수익률
- max drawdown
- win rate
- number of trades
- total fees
- equity curve

### 10.4 모의투자 API 후보

```text
GET  /paper/status
POST /paper/start
POST /paper/stop
GET  /paper/portfolio
GET  /paper/trades
POST /backtest/run
GET  /backtest/latest
```

### 10.5 대시보드 후보 화면

- 가상 초기 자본 입력
- 전략 threshold 설정
- 백테스트 실행 버튼
- 수익률/손익/거래 횟수 표시
- equity curve 차트
- 거래 내역 테이블
- 실시간 모의투자 시작/중지 버튼

## 11. API 설계

### 10.1 `GET /health`

서버 생존 확인.

응답 예시:

```json
{
  "status": "ok"
}
```

### 10.2 `GET /model/status`

모델 로드 상태 확인.

응답 예시:

```json
{
  "model_loaded": true,
  "artifact_path": "artifacts/models/current",
  "metadata": {
    "symbol": "BTCUSDT",
    "interval": "1m",
    "seq_length": 60,
    "pred_offsets": [5, 15, 30, 60]
  }
}
```

모델 없음 응답 예시:

```json
{
  "model_loaded": false,
  "reason": "model artifact not found"
}
```

### 10.3 `GET /predict`

쿼리 파라미터:

| 이름 | 기본값 | 설명 |
|---|---|---|
| symbol | BTCUSDT | 예측 대상 심볼 |
| interval | 1m | 캔들 간격 |

응답 예시:

```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "timestamp": "2026-04-26T18:30:00Z",
  "input": {
    "seq_length": 60,
    "last_price": 65000.0
  },
  "predictions": {
    "5m": 65020.1,
    "15m": 65110.5,
    "30m": 64880.2,
    "60m": 65200.0
  },
  "model": {
    "artifact": "artifacts/models/current",
    "trained_at": "2026-04-26T18:00:00Z"
  }
}
```

모델이 없는 경우:

```json
{
  "detail": "model_not_ready"
}
```

HTTP status:

```text
503 Service Unavailable
```

## 12. 대시보드 설계

Streamlit 대시보드는 다음 영역으로 구성한다.

1. 서버 상태
   - API 연결 여부
   - 모델 로드 여부
2. 현재 가격 요약
   - 최근 가격
   - 최근 60분 평균
3. 예측 결과
   - 5m/15m/30m/60m 예측 테이블
   - 최근 가격 + 예측값 라인 차트
4. 모델 평가
   - horizon별 MAE/RMSE/MAPE
   - baseline 대비 비교
5. 경고 문구
   - 투자 조언이 아님
   - 실거래 사용 금지

## 13. 설정 설계

환경변수 후보:

| 이름 | 기본값 | 설명 |
|---|---|---|
| BTCP_SYMBOL | BTCUSDT | 기본 심볼 |
| BTCP_INTERVAL | 1m | 기본 캔들 간격 |
| BTCP_SEQ_LENGTH | 60 | 입력 sequence 길이 |
| BTCP_MODEL_DIR | artifacts/models/current | 현재 모델 경로 |
| BTCP_DATA_DIR | data | 데이터 저장 경로 |
| BTCP_API_URL | http://localhost:8000 | 대시보드에서 사용할 API URL |

향후 `btcp/config.py`에서 이 값들을 읽도록 한다.

## 14. 오류 처리 원칙

- 외부 API 호출에는 timeout을 명시한다.
- Binance 응답 schema를 검증한다.
- 모델 파일이 없어도 API 서버는 시작되어야 한다.
- 예측 불가능 상태는 500이 아니라 503으로 표현한다.
- 사용자에게 내부 stack trace를 그대로 노출하지 않는다.
- 대시보드는 API 오류를 읽기 쉬운 메시지로 표시한다.

## 15. 테스트 설계

최소 테스트:

```text
tests/
  test_preprocessor.py
  test_portfolio.py
  test_strategy.py
  test_backtester.py
  test_inference.py
  test_api.py
  test_baseline.py
  test_evaluator.py
```

검증 대상:

- 입력 sequence 길이 검증
- scaler 저장/로드 일관성
- 예측 출력 shape
- 모델 없음 상태에서 `/health` 정상
- 모델 없음 상태에서 `/predict` 503
- baseline 출력 shape
- 평가 metric 계산 정확성

## 16. 개발 원칙

- 작은 단위로 변경한다.
- 먼저 실행 가능성을 확보한다.
- 모델 성능보다 평가 신뢰성을 우선한다.
- 실제 거래 기능은 검증 전까지 만들지 않는다.
- 추론과 학습의 전처리 규칙은 반드시 동일해야 한다.
- 모든 API 응답은 스키마가 예측 가능해야 한다.
- README만 보고 로컬 실행이 가능해야 한다.

## 17. 향후 전면 개편 기준

다음 조건 중 2개 이상이 충족되면 전면 개편을 검토한다.

- baseline 대비 유의미한 성능 개선 확인
- 백테스트 기준 의미 있는 전략 가능성 확인
- 여러 모델/심볼/기간을 비교해야 하는 요구 발생
- 대시보드를 장기 운영 서비스로 확장해야 함
- 데이터 저장/실험 관리가 현재 구조로 감당하기 어려워짐

전면 개편 시에는 `apps/`, `packages/`, `pipelines/`, `artifacts/` 중심 구조로 재설계한다.
