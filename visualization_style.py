# visualization_style.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def set_custom_style():
    """Set matplotlib to match your website's Brilliant.org-inspired aesthetic"""

    # Brilliant.org-inspired style with dark theme
    rcParams.update({
        # Figure aesthetics
        'figure.facecolor': '#0F172A',  # --bg-dark
        'figure.edgecolor': '#0F172A',
        'figure.figsize': (10, 5),
        'figure.dpi': 120,

        # Axes aesthetics
        'axes.facecolor': '#1E293B',  # --bg-card
        'axes.edgecolor': '#334155',  # Subtle border
        'axes.linewidth': 1,
        'axes.labelcolor': '#94A3B8',  # --text-secondary
        'axes.titlesize': 18,
        'axes.labelsize': 13,
        'axes.titleweight': 'bold',
        'axes.titlecolor': '#F1F5F9',  # --text-primary
        'axes.grid': True,
        'axes.grid.axis': 'both',
        'axes.grid.which': 'major',

        # Text aesthetics
        'text.color': '#F1F5F9',
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'DejaVu Sans', 'Arial'],

        # Tick aesthetics
        'xtick.color': '#94A3B8',
        'ytick.color': '#94A3B8',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.major.width': 1,
        'ytick.major.width': 1,

        # Grid aesthetics
        'grid.color': '#334155',
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.7,

        # Legend aesthetics
        'legend.facecolor': '#1E293B',
        'legend.edgecolor': '#475569',
        'legend.fontsize': 11,
        'legend.labelcolor': '#F1F5F9',
        'legend.framealpha': 0.9,
        'legend.frameon': True,

        # Line aesthetics
        'lines.linewidth': 2.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1,

        # Patch aesthetics (for bars, boxes, etc.)
        'patch.edgecolor': '#334155',
        'patch.linewidth': 1,
    })


# Your exact website color palette
COLORS = {
    # Primary colors
    'primary-blue': '#3B82F6',
    'primary-purple': '#8B5CF6',
    'primary-pink': '#EC4899',
    'primary-cyan': '#06B6D4',
    'primary-green': '#10B981',
    'primary-yellow': '#F59E0B',
    'primary-red': '#EF4444',

    # Text colors
    'text-primary': '#F1F5F9',
    'text-secondary': '#94A3B8',
    'text-light': '#CBD5E1',

    # Background colors
    'bg-dark': '#0F172A',
    'bg-card': '#1E293B',
    'bg-light': '#F8FAFC',

    # Additional accent colors
    'accent-teal': '#14B8A6',
    'accent-indigo': '#6366F1',
    'accent-rose': '#F43F5E',
}

# Predefined color palettes for different plot types
PALETTES = {
    'signal-noise': [COLORS['primary-blue'], COLORS['primary-green']],
    'memory': [COLORS['primary-purple'], COLORS['primary-cyan'], COLORS['primary-yellow']],
    'correlation': [COLORS['primary-pink'], COLORS['primary-blue'], COLORS['primary-green'], COLORS['primary-purple']]
}


def add_glow_effect(ax, line, color, intensity=0.3):
    """Add a subtle glow effect to lines for a more vibrant look"""
    line.set_path_effects([
        plt.matplotlib.patheffects.withStroke(linewidth=line.get_linewidth() + 3,
                                              alpha=intensity,
                                              foreground=color),
        plt.matplotlib.patheffects.Normal()
    ])
    return line


def create_gradient_line(x, y, color1, color2, ax=None, linewidth=3, alpha=0.9):
    """Create a line with a gradient color effect"""
    if ax is None:
        ax = plt.gca()

    # Simple implementation: create multiple lines with slightly different colors
    n_segments = 10
    segment_length = len(x) // n_segments

    for i in range(n_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(x))

        if start >= end:
            continue

        # Interpolate color
        t = i / (n_segments - 1) if n_segments > 1 else 0
        r = int(color1[1:3], 16) * (1 - t) + int(color2[1:3], 16) * t
        g = int(color1[3:5], 16) * (1 - t) + int(color2[3:5], 16) * t
        b = int(color1[5:7], 16) * (1 - t) + int(color2[5:7], 16) * t
        color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'

        ax.plot(x[start:end], y[start:end],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle='round')

    return ax


def add_modern_grid(ax):
    """Add a modern, subtle grid"""
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.7, color=COLORS['text-secondary'])

    # Add subtle horizontal lines at major points
    ylim = ax.get_ylim()
    for y in np.arange(np.floor(ylim[0]), np.ceil(ylim[1]) + 1):
        if y != 0:
            ax.axhline(y=y, alpha=0.1, color=COLORS['text-secondary'], linestyle='-', linewidth=0.5)

    return ax


def style_plot_area(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to plot area"""
    if title:
        ax.set_title(title, pad=20, fontsize=18, fontweight='bold', color=COLORS['text-primary'])

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13, color=COLORS['text-secondary'], labelpad=10)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, color=COLORS['text-secondary'], labelpad=10)

    # Style spines
    for spine in ax.spines.values():
        spine.set_color(COLORS['text-secondary'])
        spine.set_alpha(0.5)
        spine.set_linewidth(1)

    return ax


def save_plot(fig, path, transparent_bg=False):
    """Save plot with optimized settings for web display"""
    fig.savefig(
        path,
        dpi=120,
        facecolor=fig.get_facecolor(),
        edgecolor='none',
        bbox_inches='tight',
        pad_inches=0.1,
        transparent=transparent_bg
    )