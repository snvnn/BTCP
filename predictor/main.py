from data_collector import collect_historical_1m_data

if __name__ == '__main__':
    df = collect_historical_1m_data(
        symbol='BTCUSDT',
        start_date='2017-01-01'  # 여기를 더 과거로 설정하면 더 많이 수집됨
    )
    print(df.tail())
