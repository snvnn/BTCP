import streamlit as st
import requests
import pandas as pd

# FastAPI 백엔드 API 주소
API_URL = "http://localhost:8000/predict"  # 서버 외부에서 접속할 경우 IP:8000으로 수정

# 페이지 기본 설정
st.set_page_config(page_title="BTC 예측 대시보드", layout="wide")

st.title("🧠 실시간 BTC/USDT 예측 대시보드")

# 예측 요청 버튼
if st.button("예측 요청 🔄"):
    try:
        # FastAPI 서버로부터 예측 결과 요청
        response = requests.get(API_URL)
        data = response.json()

        st.success("✅ 예측 성공!")

        # 📊 예측 값 요약 테이블
        st.subheader("📋 예측 요약")
        df_summary = pd.DataFrame({
            "현재 시각": [data["timestamp"]],
            "최근 시세 평균": [round(sum(data["input_prices"]) / len(data["input_prices"]), 2)],
            "예측값 (정규화)": [round(data["predicted"], 4)],
            "예측값 (복원)": [round(data["denormalized_prediction"], 2)]
        })
        st.table(df_summary)

        # 📈 최근 10개 시세 + 예측값 차트
        st.subheader("📈 최근 시세 및 예측값")
        prices = data["input_prices"]
        predicted = data["denormalized_prediction"]

        # x축 라벨 설정 (T-9 ~ T-0, Predicted)
        x_labels = [f"T-{i}" for i in reversed(range(len(prices)))] + ["예측"]
        full_prices = prices + [predicted]

        df_chart = pd.DataFrame({
            "BTC/USDT": full_prices
        }, index=x_labels)

        st.line_chart(df_chart)

        # 🔎 정규화된 입력 벡터 확인 (디버깅용)
        with st.expander("정규화된 입력 벡터 보기"):
            st.write(data["normalized"])

    except Exception as e:
        st.error(f"❌ API 요청 실패: {e}")
