import numpy as np

def normalize(prices):
    prices = np.array(prices)
    mean = prices.mean()
    std = prices.std()

    # 표준편차가 0이면 나눗셈을 하지 않음
    if std == 0:
        normalized = np.zeros_like(prices)
    else:
        normalized = (prices - mean) / std

    return normalized, mean, std