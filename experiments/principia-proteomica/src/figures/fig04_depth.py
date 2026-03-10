"""Figure 4: Partition Depth — 3D surface and depth-entropy equivalence."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    depth_data = results.get('depth', {})
    surface = depth_data.get('depth_surface', {})
    slope_fit = depth_data.get('slope_fit', {})
    bio = depth_data.get('biological_scales', {})

    # Panel A: 3D depth surface
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    label_panel(ax_a, 'A', x=-0.05, y=1.02)
    _draw_depth_surface(ax_a, surface)

    # Panel B: Depth-entropy scatter with fit
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_depth_entropy_scatter(ax_b, slope_fit)

    # Panel C: Biological scale comparison
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_biological_scales(ax_c, bio)

    # Panel D: Depth distributions
    ax_d = fig.add_subplot(gs[0, 3])
    label_panel(ax_d, 'D')
    _draw_depth_distributions(ax_d, slope_fit)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_depth_surface(ax, surface):
    if not surface:
        ax.set_title('Depth Surface M(n, l)')
        return

    n = np.array(surface['n'])
    l = np.array(surface['l'])
    depth = np.array(surface['depth'])

    ax.scatter(n, l, depth, c=depth, cmap='viridis', s=20, alpha=0.7)
    ax.set_xlabel('n', fontsize=7)
    ax.set_ylabel('l', fontsize=7)
    ax.set_zlabel('M', fontsize=7)
    ax.set_title('Partition Depth M(n, l)')


def _draw_depth_entropy_scatter(ax, slope_fit):
    if not slope_fit:
        ax.set_title('Depth-Entropy Equivalence')
        return

    depths = np.array(slope_fit['depths'])
    entropies = np.array(slope_fit['entropies'])
    slope = slope_fit['slope']
    r_sq = slope_fit['r_squared']

    ax.scatter(depths, entropies, s=5, alpha=0.4, color=COLORS['primary'])

    x_fit = np.linspace(depths.min(), depths.max(), 100)
    y_fit = slope * x_fit + slope_fit['intercept']
    ax.plot(x_fit, y_fit, '-', color=COLORS['danger'], linewidth=2,
            label=f'slope = {slope:.4f}\nln(2) = {np.log(2):.4f}\nR2 = {r_sq:.6f}')

    ax.set_xlabel('Partition Depth M')
    ax.set_ylabel('S_P / k_B')
    ax.set_title('S_P = k_B M ln(b)')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3)


def _draw_biological_scales(ax, bio):
    if not bio:
        ax.set_title('Biological Scales')
        return

    scales = bio.get('scales', [])
    names = [s['name'] for s in scales]
    depths = [s['depth'] for s in scales]
    scale_types = [s['scale'] for s in scales]

    scale_colors = {
        'atomic': COLORS['primary'],
        'molecular': COLORS['secondary'],
        'protein': COLORS['tertiary'],
        'complex': COLORS['success'],
        'cellular': COLORS['danger'],
    }

    colors = [scale_colors.get(s, COLORS['neutral']) for s in scale_types]
    bars = ax.barh(names, depths, color=colors, alpha=0.8, edgecolor='white')

    for bar, d in zip(bars, depths):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(d), va='center', fontsize=7)

    ax.set_xlabel('Partition Depth M')
    ax.set_title('Depth Across Scales')
    ax.invert_yaxis()


def _draw_depth_distributions(ax, slope_fit):
    if not slope_fit:
        ax.set_title('Depth Distribution')
        return

    depths = np.array(slope_fit.get('depths', []))
    if len(depths) == 0:
        return

    ax.hist(depths, bins=20, color=COLORS['primary'], alpha=0.7,
            edgecolor='white')
    ax.axvline(np.mean(depths), color=COLORS['danger'], linestyle='--',
               linewidth=1.5, label=f'Mean = {np.mean(depths):.2f}')

    ax.set_xlabel('Partition Depth M')
    ax.set_ylabel('Count')
    ax.set_title('Depth Distribution')
    ax.legend(fontsize=7)
