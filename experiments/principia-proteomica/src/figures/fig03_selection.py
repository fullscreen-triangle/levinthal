"""Figure 3: Selection Rules — Transition matrix and enforcement ratio."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    partition = results.get('partition', {})
    trans = partition.get('transition_matrix', {})
    enforcement = partition.get('enforcement', {})

    # Panel A: Transition matrix heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    label_panel(ax_a, 'A')
    _draw_transition_matrix(ax_a, partition)

    # Panel B: Enforcement ratio
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_enforcement_ratio(ax_b, enforcement)

    # Panel C: 3D allowed transitions in state space
    ax_c = fig.add_subplot(gs[0, 2], projection='3d')
    label_panel(ax_c, 'C', x=-0.05, y=1.02)
    _draw_3d_transitions(ax_c)

    # Panel D: Allowed fraction by shell
    ax_d = fig.add_subplot(gs[0, 3])
    label_panel(ax_d, 'D')
    _draw_shell_fractions(ax_d, trans)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_transition_matrix(ax, partition_data):
    from ..core.partition import enumerate_all_states, is_allowed_transition

    states = enumerate_all_states(2)
    n = len(states)
    matrix = np.zeros((n, n))

    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if i != j and is_allowed_transition(s1, s2):
                matrix[i, j] = 1

    cmap = plt.cm.colors.ListedColormap([COLORS['danger'], COLORS['allowed']])
    ax.imshow(matrix, cmap=cmap, aspect='equal', interpolation='nearest')

    labels = [f'({s.n},{s.l},{s.m})' for s in states]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title('Transition Matrix (n<=2)')
    ax.set_xlabel('Final state')
    ax.set_ylabel('Initial state')


def _draw_enforcement_ratio(ax, enforcement):
    categories = ['Heisenberg\nlimit', 'Forbidden\n(tunneling)', 'Allowed']
    log_ratio = enforcement.get('log10_ratio', 8)
    rates = [1e12, enforcement.get('gamma_forbidden', 1e4), 1e12]
    log_rates = [np.log10(r) for r in rates]

    colors = [COLORS['neutral'], COLORS['danger'], COLORS['success']]
    bars = ax.bar(categories, log_rates, color=colors, alpha=0.8, edgecolor='white')

    for bar, lr in zip(bars, log_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'10^{lr:.0f}', ha='center', fontsize=7)

    ax.set_ylabel('log10(rate)')
    ax.set_title(f'Enforcement > 10^8')
    ax.axhline(y=0, color='black', linewidth=0.5)


def _draw_3d_transitions(ax):
    """3D visualization of allowed transitions between states."""
    from ..core.partition import enumerate_all_states, is_allowed_transition

    states = enumerate_all_states(3)

    # Plot all states
    for s in states:
        color = {1: COLORS['shell_1'], 2: COLORS['shell_2'],
                 3: COLORS['shell_3']}.get(s.n, COLORS['neutral'])
        ax.scatter(s.n, s.l, s.m, c=color, s=25, alpha=0.6)

    # Draw allowed transitions as lines
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if i < j and is_allowed_transition(s1, s2):
                ax.plot([s1.n, s2.n], [s1.l, s2.l], [s1.m, s2.m],
                        '-', color=COLORS['success'], alpha=0.15, linewidth=0.5)

    ax.set_xlabel('n', fontsize=7)
    ax.set_ylabel('l', fontsize=7)
    ax.set_zlabel('m', fontsize=7)
    ax.set_title('Allowed Transitions')


def _draw_shell_fractions(ax, trans_data):
    """Bar chart of allowed transition fraction by shell pair."""
    from ..core.partition import enumerate_all_states, is_allowed_transition

    shell_pairs = [(1, 2), (2, 3), (1, 3)]
    labels = ['1->2', '2->3', '1->3']
    fractions = []

    for n1, n2 in shell_pairs:
        states_1 = [s for s in enumerate_all_states(max(n1, n2)) if s.n == n1]
        states_2 = [s for s in enumerate_all_states(max(n1, n2)) if s.n == n2]
        total = len(states_1) * len(states_2)
        allowed = sum(1 for s1 in states_1 for s2 in states_2
                      if is_allowed_transition(s1, s2))
        fractions.append(allowed / total * 100 if total > 0 else 0)

    colors = [COLORS['shell_1'], COLORS['shell_2'], COLORS['shell_3']]
    bars = ax.bar(labels, fractions, color=colors, alpha=0.8, edgecolor='white')

    for bar, f in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{f:.1f}%', ha='center', fontsize=7)

    ax.set_xlabel('Shell Transition')
    ax.set_ylabel('Allowed (%)')
    ax.set_title('Allowed Fraction by Shell')
    ax.grid(True, alpha=0.3, axis='y')
