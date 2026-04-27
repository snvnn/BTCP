import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("BTCP_API_URL", "http://localhost:8000").rstrip("/")


def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, params: dict | None = None):
    response = requests.post(f"{API_BASE_URL}{path}", params=params or {}, timeout=10)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="BTCP 대시보드", layout="wide")
st.title("🧠 BTCP BTC/USDT 예측 · 모의투자 대시보드")
st.caption("예측 결과와 모의투자는 투자 조언이 아니며, 실거래 주문을 실행하지 않습니다.")

with st.sidebar:
    st.header("API 설정")
    st.write(f"API: `{API_BASE_URL}`")
    if st.button("서버 상태 확인"):
        try:
            st.success(api_get("/health"))
        except Exception as exc:
            st.error(f"API 연결 실패: {exc}")

prediction_tab, paper_tab = st.tabs(["가격 예측", "모의투자"])

with prediction_tab:
    st.subheader("실시간 예측")
    if st.button("예측 요청 🔄"):
        try:
            data = api_get("/predict")
            st.success("✅ 예측 성공")

            input_info = data.get("input", {})
            predictions = data.get("predictions", {})
            summary = pd.DataFrame(
                {
                    "항목": ["심볼", "인터벌", "시각", "최근 가격", "입력 길이"],
                    "값": [
                        data.get("symbol"),
                        data.get("interval"),
                        data.get("timestamp"),
                        input_info.get("last_price"),
                        input_info.get("seq_length"),
                    ],
                }
            )
            st.table(summary)

            if predictions:
                pred_df = pd.DataFrame(
                    [{"horizon": horizon, "predicted_price": price} for horizon, price in predictions.items()]
                )
                st.subheader("horizon별 예측")
                st.table(pred_df)
                st.line_chart(pred_df.set_index("horizon"))

            model_info = data.get("model", {})
            if model_info:
                with st.expander("모델 metadata"):
                    st.table(
                        pd.DataFrame(
                            {
                                "항목": ["artifact", "trained_at"],
                                "값": [model_info.get("artifact"), model_info.get("trained_at")],
                            }
                        )
                    )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 503:
                st.warning("모델이 아직 준비되지 않았습니다. 학습 artifact를 먼저 생성하세요.")
            else:
                st.error(f"예측 요청 실패: {exc}")
        except Exception as exc:
            st.error(f"예측 요청 실패: {exc}")

with paper_tab:
    st.subheader("실시간 모의투자")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        initial_cash = st.number_input("초기 가상 현금", min_value=0.0, value=10_000.0, step=100.0)
    with col2:
        threshold = st.number_input("매매 threshold", min_value=0.0, value=0.01, step=0.001, format="%.4f")
    with col3:
        trade_fraction = st.number_input("거래 비율", min_value=0.01, max_value=1.0, value=0.5, step=0.05)
    with col4:
        fee_rate = st.number_input("수수료율", min_value=0.0, value=0.001, step=0.0001, format="%.4f")

    start_col, stop_col, refresh_col = st.columns(3)
    with start_col:
        if st.button("모의투자 시작"):
            try:
                result = api_post(
                    "/paper/start",
                    params={
                        "initial_cash": initial_cash,
                        "threshold": threshold,
                        "trade_fraction": trade_fraction,
                        "fee_rate": fee_rate,
                    },
                )
                st.success("모의투자를 시작했습니다.")
                st.json(result)
            except Exception as exc:
                st.error(f"시작 실패: {exc}")
    with stop_col:
        if st.button("모의투자 중지"):
            try:
                st.json(api_post("/paper/stop"))
            except Exception as exc:
                st.error(f"중지 실패: {exc}")
    with refresh_col:
        if st.button("상태 새로고침"):
            try:
                st.json(api_get("/paper/status"))
            except Exception as exc:
                st.error(f"상태 조회 실패: {exc}")

    st.divider()
    st.subheader("수동 tick 입력")
    tick_col1, tick_col2 = st.columns(2)
    with tick_col1:
        current_price = st.number_input("현재 가격", min_value=0.0, value=100.0, step=1.0)
    with tick_col2:
        predicted_price = st.number_input("예측 가격", min_value=0.0, value=102.0, step=1.0)

    if st.button("tick 처리"):
        try:
            result = api_post(
                "/paper/tick",
                params={"current_price": current_price, "predicted_price": predicted_price},
            )
            st.json(result)
        except Exception as exc:
            st.error(f"tick 처리 실패: {exc}")

    st.divider()
    st.subheader("거래 내역")
    if st.button("거래 내역 불러오기"):
        try:
            trades = api_get("/paper/trades").get("trades", [])
            if trades:
                st.dataframe(pd.DataFrame(trades), use_container_width=True)
            else:
                st.info("아직 거래 내역이 없습니다.")
        except Exception as exc:
            st.error(f"거래 내역 조회 실패: {exc}")
