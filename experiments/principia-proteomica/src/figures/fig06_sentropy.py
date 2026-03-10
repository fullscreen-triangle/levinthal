"""Figure 6: S-Entropy Conservation — 3D simplex trajectory and backaction."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    sentropy = results.get('sentropy', {})
    measurement = sentropy.get('measurement_sequence', {})
    backaction = sentropy.get('backaction_comparison', {})

    # Panel A: 3D simplex trajectory (Sk, St, Se as axes)
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    label_panel(ax_a, 'A', x=-0.05, y=1.02)
    _draw_3d_simplex(ax_a, measurement)

    # Panel B: Component evolution over steps
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_component_evolution(ax_b, measurement)

    # Panel C: Conservation line
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_conservation(ax_c, measurement)

    # Panel D: Backaction comparison
    ax_d = fig.add_subplot(gs[0, 3])
    label_panel(ax_d, 'D')
    _draw_backaction_comparison(ax_d, backaction)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_3d_simplex(ax, measurement):
    """3D scatter of (Sk, St, Se) trajectory on the unit simplex."""
    trajectory = measurement.get('trajectory', [])
    if not trajectory:
        ax.set_title('S-Entropy Simplex')
        return

    traj = np.array(trajectory)
    n = len(traj)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=colors, s=15, alpha=0.8)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', color=COLORS['neutral'],
            linewidth=0.5, alpha=0.4)

    # Mark start and end
    ax.scatter(*traj[0], s=60, c='green', marker='o', zorder=5)
    ax.scatter(*traj[-1], s=60, c='red', marker='s', zorder=5)

    # Draw simplex boundary plane (Sk+St+Se=1)
    xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    zz = 1 - xx - yy
    mask = (zz >= 0) & (xx >= 0) & (yy >= 0)
    zz[~mask] = np.nan
    ax.plot_surface(xx, yy, zz, alpha=0.05, color=COLORS['primary'])

    ax.set_xlabel('S_k', fontsize=7)
    ax.set_ylabel('S_t', fontsize=7)
    ax.set_zlabel('S_e', fontsize=7)
    ax.set_title('S-Entropy Trajectory')


def _draw_component_evolution(ax, measurement):
    """Line plot of Sk, St, Se over measurement steps."""
    trajectory = measurement.get('trajectory', [])
    if not trajectory:
        ax.set_title('Component Evolution')
        return

    traj = np.array(trajectory)
    steps = range(len(traj))

    ax.plot(steps, traj[:, 0], '-', color=COLORS['primary'], linewidth=1.5,
            label='S_k')
    ax.plot(steps, traj[:, 1], '-', color=COLORS['secondary'], linewidth=1.5,
            label='S_t')
    ax.plot(steps, traj[:, 2], '-', color=COLORS['tertiary'], linewidth=1.5,
            label='S_e')

    ax.set_xlabel('Measurement Step')
    ax.set_ylabel('S-Entropy Component')
    ax.set_title('Component Evolution')
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def _draw_conservation(ax, measurement):
    trajectory = measurement.get('trajectory', [])
    if not trajectory:
        ax.set_title('Conservation')
        return

    totals = [sum(p) for p in trajectory]
    steps = range(len(totals))

    ax.plot(steps, totals, 'o-', color=COLORS['primary'], markersize=4, linewidth=1.5)
    ax.axhline(1.0, color=COLORS['success'], linestyle='--', linewidth=1,
               label='S_k + S_t + S_e = 1')
    ax.fill_between(steps, 0.997, 1.003, alpha=0.2, color=COLORS['success'],
                    label='+/-0.003')

    mean_total = measurement.get('mean_total', np.mean(totals))
    std_total = measurement.get('std_total', np.std(totals))

    ax.set_xlabel('Measurement Step')
    ax.set_ylabel('S_k + S_t + S_e')
    ax.set_title(f'Conservation: {mean_total:.3f} +/- {std_total:.3f}')
    ax.set_ylim(0.99, 1.01)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)


def _draw_backaction_comparison(ax, backaction):
    if not backaction:
        ax.set_title('Backaction Comparison')
        return

    categories = ['Heisenberg\nlimit', 'QND', 'Categorical']
    deltas = [
        backaction.get('heisenberg_limit', {}).get('delta', 1.0),
        backaction.get('qnd', {}).get('delta', 1e-3),
        backaction.get('categorical', {}).get('delta', 1.68e-4),
    ]
    log_deltas = [np.log10(d) for d in deltas]

    colors = [COLORS['danger'], COLORS['secondary'], COLORS['success']]
    bars = ax.bar(categories, log_deltas, color=colors, alpha=0.8, edgecolor='white')

    for bar, d, ld in zip(bars, deltas, log_deltas):
        ax.text(bar.get_x() + bar.get_width()/2, ld - 0.2,
                f'{d:.0e}', ha='center', fontsize=7, color='white', fontweight='bold')

    ax.set_ylabel('log10(delta)')
    ax.set_title('Measurement Backaction')
    ax.grid(True, alpha=0.3, axis='y')
