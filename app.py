import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 1. 제목 표시
st.title("Bike Sharing 시계열 예측")
st.write("월별 자전거 대여량을 확인하고 미래 대여량을 예측합니다.")

# 2. CSV 파일 업로드
uploaded_file = st.file_uploader("Bike_Sharing_Demand.csv 파일을 업로드하세요.", type=["csv"])

if uploaded_file is not None:
    # 3. 데이터 불러오기
    df = pd.read_csv(uploaded_file)
    
    # 4. datetime 자료형 변환
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # 5. 시간 순서 정렬 및 월별 count 합계 계산
    df = df.sort_values(by="datetime")
    df["year_month"] = df["datetime"].dt.to_period("M").dt.to_timestamp()
    monthly_data = df.groupby("year_month")["count"].sum().to_frame()
    
    # 6. 월별 데이터 표와 선 그래프 표시
    st.subheader("과거 월별 대여량 데이터")
    st.dataframe(monthly_data)
    st.line_chart(monthly_data["count"])
    
    # 7. 예측 기간 선택 (1, 3, 6개월)
    forecast_period = st.selectbox(
        "예측 기간(월)을 선택하세요:",
        options=[1, 3, 6]
    )
    
    # 8. 전체 월별 데이터로 ARIMA(2, 1, 1) 학습
    model = ARIMA(monthly_data["count"], order=(2, 1, 1))
    model_fit = model.fit()
    
    # 9. 선택 기간만큼 미래 예측
    forecast_values = model_fit.forecast(steps=forecast_period)
    
    # 10. 마지막 관측 월의 다음 달부터 미래 날짜 인덱스 생성
    last_date = monthly_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_period, freq="MS")
    
    # 11. 미래 날짜와 예측값 DataFrame 생성 및 표 표시
    forecast_df = pd.DataFrame({
        "예측 날짜": future_dates,
        "예측 대여량": forecast_values.values
    }).set_index("예측 날짜")
    
    st.subheader(f"향후 {forecast_period}개월 대여량 예측 결과")
    st.dataframe(forecast_df.style.format({"예측 대여량": "{:,.0f}"}))
    
    # 12. 기존 실제값과 미래 예측값을 하나의 그래프로 표시
    # 두 시계열을 병합(concat)하여 컬럼으로 구분
    combined_df = pd.DataFrame({
        "실제 대여량": monthly_data["count"],
        "미래 예측값": pd.Series(forecast_values.values, index=future_dates)
    })
    
    st.subheader("실제 대여량과 미래 예측값 비교 차트")
    st.line_chart(combined_df)

else:
    st.info("시계열 분석 및 미래 예측을 위해 CSV 파일을 업로드해주세요.")