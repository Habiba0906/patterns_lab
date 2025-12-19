import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_time_series(n=300, memory_strength=0.8, seed=42):

    np.random.seed(seed)

    noise = np.random.normal(0, 1, n)
    series = np.zeros(n)

    for t in range(1, n):
        series[t] = memory_strength * series[t - 1] + noise[t]

    return pd.Series(series)


def plot_temporal_memory(series, window=20, save_path=None):

    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()

    plt.figure(figsize=(8, 4))
    plt.plot(series, label="Observed series", linewidth=1)
    plt.plot(rolling_mean, label="Rolling mean", linewidth=2)
    plt.plot(rolling_std, label="Rolling std", linestyle="--", linewidth=1)

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
