import numpy as np

def normalize(prices):
    prices = np.array(prices)
    mean = prices.mean()
    std = prices.std()
    return (prices - mean) / std, mean, std
