from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')  # Important for server environments

# Import your visualization style module
from visualization_style import set_custom_style, COLORS, PALETTES, style_plot_area, add_modern_grid, save_plot

# Your existing imports
from analysis.memory import generate_time_series, plot_temporal_memory
from analysis.correlation import generate_independent_series, plot_correlation_experiment

app = Flask(__name__)


# ========== SIGNAL & NOISE ==========
@app.route("/noise")
def noise():
    level = float(request.args.get("level", 0.3))

    # Generate data
    x = np.linspace(0, 10, 250)  # More points for smoother curves
    signal = np.sin(x) * 0.8 + 0.2 * np.sin(2.5 * x)  # More interesting signal
    noise_values = np.random.normal(0, level, size=len(x))
    y = signal + noise_values

    # Set custom style
    set_custom_style()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Plot with enhanced styling
    observed_line, = ax.plot(x, y,
                             color=COLORS['primary-blue'],
                             linewidth=2.5,
                             alpha=0.85,
                             label='Observed (Signal + Noise)',
                             marker='o',
                             markersize=4,
                             markevery=15,
                             markerfacecolor='white',
                             markeredgecolor=COLORS['primary-blue'],
                             markeredgewidth=1.5)

    signal_line, = ax.plot(x, signal,
                           color=COLORS['primary-green'],
                           linewidth=3.5,
                           alpha=1,
                           label='True Signal',
                           linestyle='-',
                           solid_capstyle='round')

    # Add fill between for noise visualization
    ax.fill_between(x, signal, y,
                    color=COLORS['primary-blue'],
                    alpha=0.15,
                    label='Noise Amplitude')

    # Style the plot
    ax = style_plot_area(ax,
                         title=f'Signal vs Noise • Level: {level:.2f}',
                         xlabel='Time',
                         ylabel='Value')

    # Add modern grid
    ax = add_modern_grid(ax)

    # Add annotation about noise level
    ax.annotate(f'Noise σ = {level:.2f}',
                xy=(0.02, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                color=COLORS['text-secondary'],
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor=COLORS['bg-card'],
                          edgecolor=COLORS['primary-blue'],
                          alpha=0.8))

    # Enhanced legend
    legend = ax.legend(loc='upper right',
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       borderpad=1)
    legend.get_frame().set_alpha(0.9)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    fig_path = "static/figures/noise.png"
    save_plot(fig, fig_path)
    plt.close(fig)

    return render_template("noise.html", level=level)


@app.route("/memory")
def memory():
    strength = float(request.args.get("strength", 0.8))
    window = int(request.args.get("window", 20))

    # Generate data
    np.random.seed(42)
    n_points = 200
    series = np.zeros(n_points)
    series[0] = np.random.randn()

    for i in range(1, n_points):
        series[i] = strength * series[i - 1] + np.random.randn() * (1 - strength)

    # Add some trend for visual interest
    trend = np.linspace(0, 2, n_points)
    series = series + trend * 0.3

    # Set custom style
    set_custom_style()

    # CREATE WIDER FIGURE - Increased from 13 to 15 inches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),  # WIDER: 15 instead of 13
                                   gridspec_kw={'width_ratios': [2, 1]})

    # Plot main time series
    plot_series = series[:window]
    time_indices = np.arange(window)

    main_line, = ax1.plot(time_indices, plot_series,
                          color=COLORS['primary-purple'],
                          linewidth=3,
                          alpha=0.9,
                          label=f'Time Series (α={strength:.2f})',
                          marker='o',
                          markersize=5,
                          markevery=max(1, window // 10),
                          markerfacecolor='white',
                          markeredgecolor=COLORS['primary-purple'])

    # Calculate and plot moving average
    rolling_window = min(5, window)
    if rolling_window > 1 and window > rolling_window:
        rolling_avg = np.convolve(plot_series,
                                  np.ones(rolling_window) / rolling_window,
                                  mode='valid')
        rolling_time = time_indices[rolling_window - 1:]

        ax1.plot(rolling_time, rolling_avg,
                 color=COLORS['primary-cyan'],
                 linewidth=2.5,
                 alpha=0.8,
                 linestyle='--',
                 label=f'{rolling_window}-point Moving Avg')

    # Style main plot - Add more padding
    ax1 = style_plot_area(ax1,
                          title=f'Temporal Memory • Strength: {strength:.2f} • Window: {window} points',
                          xlabel='Time Step',
                          ylabel='Value')

    # Adjust x-axis limits to prevent cutting
    ax1.set_xlim(-1, window + 1)  # Add padding on both sides

    ax1 = add_modern_grid(ax1)

    # Position legend better
    ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))

    # Autocorrelation plot
    max_lag = min(30, window // 2)
    if max_lag > 1:
        autocorr = np.correlate(plot_series, plot_series, mode='full')
        autocorr = autocorr[autocorr.size // 2:autocorr.size // 2 + max_lag]
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]

        lags = np.arange(max_lag)
        bars = ax2.bar(lags, autocorr,
                       color=COLORS['primary-pink'],
                       alpha=0.7,
                       edgecolor=COLORS['primary-pink'],
                       linewidth=1)

        # Highlight first few lags
        highlight_count = min(5, max_lag)
        for i in range(highlight_count):
            bars[i].set_alpha(1)
            bars[i].set_color(COLORS['primary-yellow'])

        # Style autocorrelation plot
        ax2 = style_plot_area(ax2,
                              title=f'Autocorrelation (max lag: {max_lag})',
                              xlabel='Lag',
                              ylabel='Correlation')
        ax2 = add_modern_grid(ax2)
        ax2.axhline(y=0, color=COLORS['text-secondary'], linestyle='-', alpha=0.5)

        # Set reasonable x-ticks
        if max_lag > 5:
            ax2.set_xticks(np.arange(0, max_lag, max(1, max_lag // 5)))
        else:
            ax2.set_xticks(np.arange(max_lag))
    else:
        ax2.text(0.5, 0.5, 'Window too small\nfor autocorrelation',
                 ha='center', va='center',
                 transform=ax2.transAxes,
                 fontsize=12,
                 color=COLORS['text-secondary'])
        ax2.axis('off')

    # Add insight text with better positioning
    if window > 1:
        correlation = np.corrcoef(plot_series[:-1], plot_series[1:])[0, 1]
        insight_text = f"• Memory strength: {strength:.2f}\n"
        insight_text += f"• Observation window: {window} points\n"
        insight_text += f"• 1-lag correlation: {correlation:.3f}\n"
        insight_text += f"• Data shows {'strong' if correlation > 0.7 else 'moderate' if correlation > 0.3 else 'weak'} persistence"

    fig.text(0.02, 0.02, insight_text,
             fontsize=11,  # Slightly larger
             color=COLORS['text-secondary'],
             bbox=dict(boxstyle='round,pad=0.6',
                       facecolor=COLORS['bg-card'],
                       edgecolor=COLORS['primary-purple'],
                       alpha=0.8))

    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)  # Increased padding

    # Save figure
    fig_path = "static/figures/memory.png"
    save_plot(fig, fig_path)
    plt.close(fig)

    return render_template("memory.html", strength=strength, window=window)

# ========== STRUCTURAL CORRELATION ==========
@app.route("/correlation")
def correlation():
    # REASONABLE DEFAULTS:
    memory = float(request.args.get("memory", 0.8))  # Was 0.8
    trend = float(request.args.get("trend", 0.01))  # Was 0.01 (not 0.250)
    noise_level = float(request.args.get("noise", 1.0))  # Was 1.0 (not 1.400)
    window = int(request.args.get("window", 30))  # Was 30 (not 300)


    # Generate enhanced correlated series
    np.random.seed(42)
    n_points = 200
    t = np.linspace(0, 10, n_points)

    # Create two independent processes with shared characteristics
    # Process A
    a = np.zeros(n_points)
    a[0] = np.random.randn()
    for i in range(1, n_points):
        a[i] = memory * a[i - 1] + np.random.randn() * (1 - memory)
    a = a + trend * t * 5 + np.random.randn(n_points) * noise_level * 0.5

    # Process B (independent but with similar structure)
    b = np.zeros(n_points)
    b[0] = np.random.randn()
    for i in range(1, n_points):
        b[i] = memory * b[i - 1] + np.random.randn() * (1 - memory)
    b = b + trend * t * 3 + np.random.randn(n_points) * noise_level * 0.7

    # Apply smoothing window
    kernel = np.ones(window) / window
    a_smooth = np.convolve(a, kernel, mode='valid')
    b_smooth = np.convolve(b, kernel, mode='valid')
    t_smooth = t[window // 2:-(window // 2)] if window % 2 == 0 else t[window // 2:-(window // 2 - 1)]

    # Calculate correlation
    correlation = np.corrcoef(a_smooth[:len(t_smooth)], b_smooth[:len(t_smooth)])[0, 1]

    # Set custom style
    set_custom_style()

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 10))

    # Grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)

    # Top: Both series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, a,
             color=COLORS['primary-blue'],
             linewidth=2.5,
             alpha=0.8,
             label='Process A')
    ax1.plot(t, b,
             color=COLORS['primary-green'],
             linewidth=2.5,
             alpha=0.8,
             label='Process B')

    # Add smoothed versions
    ax1.plot(t_smooth, a_smooth[:len(t_smooth)],
             color=COLORS['primary-blue'],
             linewidth=4,
             alpha=0.4,
             label=f'A (Smoothed, w={window})')
    ax1.plot(t_smooth, b_smooth[:len(t_smooth)],
             color=COLORS['primary-green'],
             linewidth=4,
             alpha=0.4,
             label=f'B (Smoothed, w={window})')

    ax1 = style_plot_area(ax1,
                          title=f'Structural Correlation • r = {correlation:.3f}',
                          xlabel='Time',
                          ylabel='Value')
    ax1 = add_modern_grid(ax1)
    ax1.legend(ncol=2, loc='upper left')

    # Middle left: Scatter plot
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(a_smooth[:len(t_smooth)], b_smooth[:len(t_smooth)],
                          c=t_smooth,
                          cmap='viridis',
                          alpha=0.7,
                          s=30,
                          edgecolors='white',
                          linewidths=0.5)

    # Add regression line
    z = np.polyfit(a_smooth[:len(t_smooth)], b_smooth[:len(t_smooth)], 1)
    p = np.poly1d(z)
    ax2.plot(sorted(a_smooth[:len(t_smooth)]),
             p(sorted(a_smooth[:len(t_smooth)])),
             color=COLORS['primary-pink'],
             linewidth=3,
             alpha=0.8,
             label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')

    ax2 = style_plot_area(ax2,
                          title='Scatter Plot',
                          xlabel='Process A',
                          ylabel='Process B')
    ax2 = add_modern_grid(ax2)
    ax2.legend()

    # Middle right: Parameter visualization
    ax3 = fig.add_subplot(gs[1, 1])

    parameters = ['Memory', 'Trend', 'Noise', 'Window']
    values = [memory, trend, noise_level, window]
    colors_bar = [COLORS['primary-purple'], COLORS['primary-yellow'],
                  COLORS['primary-red'], COLORS['primary-cyan']]

    bars = ax3.bar(parameters, values, color=colors_bar, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{val:.3f}' if isinstance(val, float) else f'{val}',
                 ha='center', va='bottom',
                 fontsize=11,
                 color=COLORS['text-primary'])

    ax3 = style_plot_area(ax3,
                          title='Parameters',
                          xlabel='',
                          ylabel='Value')
    ax3.set_ylim(0, max(values) * 1.2)
    ax3 = add_modern_grid(ax3)

    # Bottom: Insight panel
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    insight_text = "✨ INSIGHTS ✨\n\n"
    insight_text += f"• These processes are mathematically independent\n"
    insight_text += f"• Yet they appear correlated (r = {correlation:.3f})\n"
    insight_text += f"• Why? Shared structure: memory ({memory:.2f}), trend ({trend:.3f})\n"
    insight_text += f"• Smoothing (window={window}) amplifies apparent relationship\n\n"
    insight_text += "⚠️ Correlation ≠ Causation!"

    ax4.text(0.02, 0.5, insight_text,
             fontsize=12,
             color=COLORS['text-primary'],
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=1',
                       facecolor=COLORS['bg-card'],
                       edgecolor=COLORS['primary-purple'],
                       alpha=0.9))

    plt.tight_layout()

    # Save figure
    fig_path = "static/figures/correlation.png"
    save_plot(fig, fig_path)
    plt.close(fig)

    return render_template("correlation.html",
                           memory=memory,
                           trend=trend,
                           noise=noise_level,
                           window=window)


# EXISTING ROUTES
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


# App entry point
if __name__ == "__main__":
    os.makedirs("static/figures", exist_ok=True)
    app.run(debug=True)