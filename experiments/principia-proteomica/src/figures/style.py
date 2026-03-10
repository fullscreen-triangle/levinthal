"""Unified figure style for Principia Proteomica."""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def apply_style():
    """Apply consistent publication style to all figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 8,
        'axes.linewidth': 0.8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.pad_inches': 0.1,
    })


COLORS = {
    'primary': '#2196F3',
    'secondary': '#FF9800',
    'tertiary': '#9C27B0',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FFC107',
    'neutral': '#9E9E9E',
    'dark': '#37474F',
    'copper': '#B87333',
    'zinc': '#7D7D7D',
    'shell_1': '#2196F3',
    'shell_2': '#4CAF50',
    'shell_3': '#FF9800',
    'shell_4': '#F44336',
    'allowed': '#4CAF50',
    'forbidden': '#F44336',
}

PANEL_LABELS = ['A', 'B', 'C', 'D']
FIGSIZE = (24, 5.5)
DPI = 300


def create_4panel(figsize=FIGSIZE):
    """Create a 1x4 panel figure (four charts in a row)."""
    apply_style()
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 4, figure=fig, wspace=0.28,
                  left=0.04, right=0.97, top=0.88, bottom=0.12)
    return fig, gs


def label_panel(ax, label, x=-0.08, y=1.12):
    """Add a bold panel label (A, B, C, D)."""
    try:
        ax.text2D(x, y, label, transform=ax.transAxes,
                  fontsize=12, fontweight='bold', va='top', ha='left')
    except AttributeError:
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')
