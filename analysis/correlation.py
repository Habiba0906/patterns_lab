import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def generate_independent_series(
    n=300,
    memory_strength=0.8,
    trend_strength=0.01,
    noise_level=1.0,
    seed=None,
):

    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n)

    a = np.zeros(n)
    b = np.zeros(n)

    noise_a = np.random.normal(0, noise_level, size=n)
    noise_b = np.random.normal(0, noise_level, size=n)

    for i in range(1, n):
        a[i] = (
            memory_strength * a[i - 1]
            + trend_strength * t[i]
            + noise_a[i]
        )
        b[i] = (
            memory_strength * b[i - 1]
            + trend_strength * t[i]
            + noise_b[i]
        )

    return t, a, b


def rolling_correlation(x, y, window=30):

    r = np.full(len(x), np.nan)

    for i in range(window, len(x)):
        r[i], _ = pearsonr(x[i - window : i], y[i - window : i])

    return r


def plot_correlation_experiment(
    t,
    a,
    b,
    window=30,
    save_path=None,
):

    # Global Pearson correlation
    r, _ = pearsonr(a, b)

    r_roll = rolling_correlation(a, b, window)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # --- Time series plot ---
    axes[0].plot(t, a, label="Series A", linewidth=1.5)
    axes[0].plot(t, b, label="Series B", linewidth=1.5)
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].set_title(f"Independent Series (Pearson r = {r:.2f})")

    # --- Rolling correlation plot ---
    axes[1].plot(t, r_roll)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Rolling r")
    axes[1].set_xlabel("Time")
    axes[1].set_title(f"Rolling Correlation (window = {window})")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
