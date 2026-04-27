# BTCP

BTCP는 Binance의 BTC/USDT 시세 데이터를 수집하고, 시계열 모델을 이용해 단기 가격 변화를 예측하는 Python 기반 프로젝트입니다. 현재 목표는 기존 프로토타입을 안정화하여 **작동하는 BTC 예측 API와 실시간 대시보드**를 만드는 것입니다.

> ⚠️ 이 프로젝트의 예측 결과는 투자 조언이 아닙니다. 실제 매매에 사용하기 전에 충분한 검증, 백테스트, 리스크 관리가 필요합니다.

## 현재 프로젝트 상태

이 저장소는 초기 프로토타입 단계입니다.

현재 포함된 기능:

- Binance BTC/USDT 1분봉 데이터 수집
- LSTM 모델 학습 코드
- FastAPI 기반 예측 API
- Streamlit 기반 예측 대시보드
- Dockerfile 및 docker-compose 초안
- 실시간/최근 데이터 기반 평가 코드 초안
- 향후 추가할 모의투자/paper trading 기능의 설계 방향

아직 보완이 필요한 부분:

- 학습/추론 전처리 방식 통일
- 모델 artifact 저장/로드 규칙 정리
- baseline 모델과 정량 평가 지표 추가
- 백테스트와 모의투자 기능 추가
- README 외 상세 문서/테스트/CI 보강
- Docker volume 경로와 실행 절차 검증
- 모델 파일이 없을 때의 API 동작 안정화

## 기술 스택

- Python 3.11 기준 Dockerfile
- FastAPI
- Uvicorn
- Streamlit
- Pandas / NumPy
- scikit-learn
- TensorFlow / Keras
- Binance API
- Docker / Docker Compose

## 저장소 구조

```text
BTCP/
  btcp/
    api/
      server.py              # FastAPI 예측 API
    data/
      collector.py           # Binance historical/recent kline 수집
      crawler.py             # 과거 데이터 수집 스크립트
      realtime.py            # 실시간 데이터 수집 유틸
    model/
      trainer.py             # LSTM 모델 학습
      inference.py           # 학습된 모델 로드 및 추론
      evaluator.py           # 실시간/최근 데이터 평가
    utils/
      preprocessor.py        # 전처리 유틸
    live_predictor.py        # 실시간 예측 루프 초안
    main.py                  # 데이터 수집 실행 진입점
  dashboard/
    streamlit_app.py         # Streamlit 대시보드
  docker/
    Dockerfile.api
    Dockerfile.streamlit
    docker-compose.yml
  scripts/
    startsh                  # 로컬 API 실행 메모성 스크립트
  requirements.txt
```

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/snvnn/BTCP.git
cd BTCP
```

### 2. 가상환경 생성

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 참고: TensorFlow, NumPy, Python 버전 조합에 따라 설치 문제가 생길 수 있습니다. 현재 `requirements.txt`는 안정화 전 단계이므로 추후 `pyproject.toml` 또는 고정된 lock 파일로 정리할 예정입니다.

## 실행

### FastAPI 서버

```bash
uvicorn btcp.api.server:app --host 0.0.0.0 --port 8000 --reload
```

접속 확인:

```bash
curl http://localhost:8000/
```

예측 요청:

```bash
curl http://localhost:8000/predict
```

> 현재 API는 서버 시작 시 모델 파일을 로드하려고 합니다. 모델 파일이 없거나 경로가 맞지 않으면 서버 실행에 실패할 수 있습니다. 향후 `/health`, `/model/status`, 모델 없음 상태 처리 기능을 추가할 계획입니다.

### Streamlit 대시보드

별도 터미널에서 실행합니다.

```bash
streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

브라우저에서 접속:

```text
http://localhost:8501
```

## Docker 실행 초안

```bash
cd docker
docker-compose up --build
```

서비스:

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`
- Postgres: `localhost:5432`

> 현재 compose 파일은 초안입니다. volume 경로, 모델 경로, 데이터 경로는 실제 운영 전 정리가 필요합니다.

## 데이터 수집

과거 1분봉 데이터를 수집하는 기본 진입점:

```bash
python -m btcp.main
```

또는 crawler 모듈 직접 실행:

```bash
python -m btcp.data.crawler
```

수집 대상 기본값:

- Symbol: `BTCUSDT`
- Interval: `1m`

## 모델 학습

현재 학습 스크립트:

```bash
python -m btcp.model.trainer
```

주의사항:

- 현재 `trainer.py`에는 로컬 절대 경로가 포함되어 있습니다.
- 학습 데이터 CSV 경로와 모델 저장 경로는 설정 파일 기반으로 바꿀 예정입니다.
- 학습 시 사용하는 scaler와 추론 시 사용하는 scaler를 함께 저장/로드하도록 개선해야 합니다.

## 평가

최근 데이터 기반 빠른 평가:

```bash
python -m btcp.model.evaluator --recent
```

실시간 평가:

```bash
python -m btcp.model.evaluator
```

향후 추가할 평가 지표:

- MAE
- RMSE
- MAPE
- Directional Accuracy
- naive baseline 대비 성능
- 간단한 수익률/백테스트 지표

## 모의투자 방향

BTCP는 단순히 예측값을 보여주는 도구에 머무르지 않고, 예측이 실제 의사결정에 도움이 되는지 검증하기 위해 **모의투자(paper trading)** 기능을 핵심 확장 범위에 포함합니다.

모의투자는 실제 주문을 넣지 않고 가상 잔고와 가상 포지션을 관리하면서 전략의 성과를 확인하는 기능입니다.

우선순위는 다음과 같습니다.

1. 과거 데이터 기반 백테스트
2. 가상 포트폴리오/주문/체결 모델
3. 예측값을 매수/매도/관망 신호로 바꾸는 전략 계층
4. 실시간 paper trading loop
5. 대시보드에서 가상 잔고, 포지션, 손익, 거래 내역 표시

실거래 연동은 현재 범위가 아니며, 백테스트와 모의투자로 충분히 검증된 뒤에만 검토합니다.

## 문서

- [설계서](docs/DESIGN.md)
- [절차별 개발 계획](docs/DEVELOPMENT_PLAN.md)

## 권장 개발 방향

현재 단계에서는 전면 개편보다 **기존 코드를 개선하면서 작동 가능한 데모와 검증 체계를 먼저 만드는 방향**을 채택합니다.

우선순위:

1. 문서화 및 실행 절차 정리
2. 모델 없음 상태에서도 API health check 가능하게 만들기
3. 학습/추론 전처리 통일
4. 예측 응답 스키마 정리
5. baseline 및 정량 평가 추가
6. 백테스트와 모의투자 기능 추가
7. Streamlit 대시보드 개선
8. 테스트와 CI 추가

## 라이선스

라이선스가 아직 명시되어 있지 않습니다. 공개 배포/협업 전 `LICENSE` 파일을 추가하는 것을 권장합니다.
