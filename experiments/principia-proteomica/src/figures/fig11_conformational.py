"""Figure 11: Conformational Dynamics — Loop gating and temperature independence."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    conf = results.get('conformational', {})
    gating = conf.get('loop_gating', {})
    temp = conf.get('temperature_independence', {})

    # Panel A: Bar chart of loop states
    ax_a = fig.add_subplot(gs[0, 0])
    label_panel(ax_a, 'A')
    _draw_loop_states(ax_a, gating)

    # Panel B: Gating transition r(t)
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_gating_transition(ax_b, gating)

    # Panel C: Temperature independence
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_temp_independence(ax_c, temp)

    # Panel D: 3D temperature-time-r surface
    ax_d = fig.add_subplot(gs[0, 3], projection='3d')
    label_panel(ax_d, 'D', x=-0.05, y=1.02)
    _draw_3d_temp_surface(ax_d, temp)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_loop_states(ax, gating):
    r_closed = gating.get('r_closed', 0.92)
    r_open = gating.get('r_open', 0.71)
    r_reclosed = gating.get('r_reclosed', 0.85)

    states = ['Closed', 'Open', 'Re-closed']
    r_values = [r_closed, r_open, r_reclosed]
    colors = [COLORS['success'], COLORS['danger'], COLORS['secondary']]

    bars = ax.bar(states, r_values, color=colors, alpha=0.8, edgecolor='white')
    for bar, r in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{r:.2f}', ha='center', fontsize=8, fontweight='bold')

    ax.set_ylabel('r')
    ax.set_title('Loop State Coherence')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.8, color=COLORS['neutral'], linestyle='--', linewidth=0.8)


def _draw_gating_transition(ax, gating):
    r_series = gating.get('r_series', [])
    if not r_series:
        ax.set_title('Gating Transition')
        return

    n = len(r_series)
    t = range(n)

    ax.plot(t, r_series, '-', color=COLORS['primary'], linewidth=1, alpha=0.8)

    third = n // 3
    ax.axvspan(0, third, alpha=0.1, color=COLORS['success'], label='Closed')
    ax.axvspan(third, 2*third, alpha=0.1, color=COLORS['danger'], label='Open')
    ax.axvspan(2*third, n, alpha=0.1, color=COLORS['secondary'], label='Re-close')

    ax.set_xlabel('Time step')
    ax.set_ylabel('r')
    ax.set_title('Loop Gating Dynamics')
    ax.legend(fontsize=6, loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def _draw_temp_independence(ax, temp):
    if not temp:
        ax.set_title('Temperature Independence')
        return

    results = temp.get('results', {})
    temps = []
    r_values = []

    for key, data in sorted(results.items(), key=lambda x: x[1]['temperature']):
        temps.append(data['temperature'])
        r_values.append(data['final_r'])

    ax.plot(temps, r_values, 'o-', color=COLORS['primary'], markersize=8,
            linewidth=2)

    r_spread = temp.get('r_spread', 0)
    ax.fill_between(temps, [r - 0.05 for r in r_values],
                    [r + 0.05 for r in r_values], alpha=0.2,
                    color=COLORS['primary'])

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Final r')
    ax.set_title(f'Temp Independence (spread = {r_spread:.3f})')
    ax.grid(True, alpha=0.3)


def _draw_3d_temp_surface(ax, temp):
    """3D surface: temperature x time x order parameter."""
    if not temp:
        ax.set_title('T-t-r Surface')
        return

    results = temp.get('results', {})
    if not results:
        ax.set_title('T-t-r Surface')
        return

    # Collect r_series from each temperature run
    temp_list = []
    r_matrix = []

    for key, data in sorted(results.items(), key=lambda x: x[1]['temperature']):
        temp_list.append(data['temperature'])
        r_series = data.get('r_series', [])
        if r_series:
            r_matrix.append(r_series)

    if not r_matrix:
        ax.set_title('T-t-r Surface')
        return

    # Ensure all series same length
    min_len = min(len(r) for r in r_matrix)
    r_matrix = [r[:min_len] for r in r_matrix]
    r_arr = np.array(r_matrix)

    T = np.array(temp_list)
    t = np.arange(min_len)
    TT, tt = np.meshgrid(T, t, indexing='ij')

    ax.plot_surface(TT, tt, r_arr, cmap='viridis', alpha=0.7, linewidth=0)
    ax.set_xlabel('T (K)', fontsize=7)
    ax.set_ylabel('Time', fontsize=7)
    ax.set_zlabel('r', fontsize=7)
    ax.set_title('T-t-r Surface')
