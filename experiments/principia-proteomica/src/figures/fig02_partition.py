"""Figure 2: Partition Coordinate System — 3D state space and capacity."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    partition = results.get('partition', {})
    all_states = partition.get('all_states', [])

    # Panel A: 3D state space
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    label_panel(ax_a, 'A', x=-0.05, y=1.02)
    _draw_3d_states(ax_a, all_states)

    # Panel B: C(n) = 2n² bar chart
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_capacity_bars(ax_b)

    # Panel C: Subshell capacities
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_subshell_capacities(ax_c)

    # Panel D: Cumulative shell filling
    ax_d = fig.add_subplot(gs[0, 3])
    label_panel(ax_d, 'D')
    _draw_cumulative(ax_d)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_3d_states(ax, all_states):
    shell_colors = {1: COLORS['shell_1'], 2: COLORS['shell_2'],
                    3: COLORS['shell_3'], 4: COLORS['shell_4']}

    for state in all_states:
        n, l, m, s = state['n'], state['l'], state['m'], state['s']
        color = shell_colors.get(n, COLORS['neutral'])
        marker = '^' if s > 0 else 'v'
        ax.scatter(n, l, m, c=color, marker=marker, s=30, alpha=0.7)

    ax.set_xlabel('n (depth)', fontsize=7)
    ax.set_ylabel('l (complexity)', fontsize=7)
    ax.set_zlabel('m (orientation)', fontsize=7)
    ax.set_title('Partition State Space')


def _draw_capacity_bars(ax):
    n_values = [1, 2, 3, 4]
    capacities = [2 * n**2 for n in n_values]
    colors = [COLORS['shell_1'], COLORS['shell_2'],
              COLORS['shell_3'], COLORS['shell_4']]

    bars = ax.bar(n_values, capacities, color=colors, alpha=0.8, edgecolor='white')
    for bar, cap in zip(bars, capacities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(cap), ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Shell n')
    ax.set_ylabel('Capacity C(n)')
    ax.set_title('C(n) = 2n²')
    ax.set_xticks(n_values)


def _draw_subshell_capacities(ax):
    l_values = [0, 1, 2, 3]
    labels = ['s', 'p', 'd', 'f']
    capacities = [2 * (2*l + 1) for l in l_values]
    colors = [COLORS['primary'], COLORS['secondary'],
              COLORS['tertiary'], COLORS['danger']]

    bars = ax.bar(labels, capacities, color=colors, alpha=0.8, edgecolor='white')
    for bar, cap in zip(bars, capacities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(cap), ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Subshell l')
    ax.set_ylabel('Capacity 2(2l+1)')
    ax.set_title('Subshell Capacities')


def _draw_cumulative(ax):
    n_values = range(1, 8)
    cumulative = []
    total = 0
    for n in n_values:
        total += 2 * n**2
        cumulative.append(total)

    ax.plot(list(n_values), cumulative, 'o-', color=COLORS['primary'],
            linewidth=2, markersize=6)

    noble = {1: 2, 2: 10, 3: 28, 4: 60}
    for n, c in noble.items():
        ax.annotate(f'n={n}: {c}', xy=(n, c), xytext=(n + 0.3, c + 5),
                   fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.7))

    ax.set_xlabel('Shell n')
    ax.set_ylabel('Cumulative States')
    ax.set_title('Cumulative Capacity')
    ax.grid(True, alpha=0.3)
