"""Figure 8: Electron Transfer — 3D trajectory and categorical coordinates."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    et = results.get('electron_transfer', {})
    traj = et.get('trajectory', {})
    trisection = et.get('trisection', {})
    sentropy = et.get('sentropy', {})

    # Panel A: 3D electron trajectory
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    label_panel(ax_a, 'A', x=-0.05, y=1.02)
    _draw_3d_trajectory(ax_a, traj)

    # Panel B: Coordinate transitions
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_coordinate_transitions(ax_b, trisection)

    # Panel C: S-entropy along trajectory
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_sentropy_trajectory(ax_c, sentropy)

    # Panel D: Backaction per iteration
    ax_d = fig.add_subplot(gs[0, 3])
    label_panel(ax_d, 'D')
    _draw_backaction_profile(ax_d, trisection)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_3d_trajectory(ax, traj):
    positions = traj.get('positions', [])
    if not positions:
        ax.set_title('Electron Trajectory')
        return

    positions = np.array(positions)
    n = len(positions)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    for i in range(n - 1):
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                '-', color=colors[i], linewidth=1.5, alpha=0.8)

    ax.scatter(*positions[0], s=80, c='green', marker='o', zorder=5, label='Cu(I)')
    ax.scatter(*positions[-1], s=80, c='red', marker='s', zorder=5, label='Cu(II)')

    ax.set_xlabel('x (A)', fontsize=7)
    ax.set_ylabel('y (A)', fontsize=7)
    ax.set_zlabel('z (A)', fontsize=7)
    ax.set_title('Electron Transfer Trajectory')
    ax.legend(fontsize=6)


def _draw_coordinate_transitions(ax, trisection):
    trits = trisection.get('trit_sequence', [])
    resolutions = trisection.get('resolutions', [])

    if not trits:
        ax.set_title('Trisection Iterations')
        return

    iterations = range(1, len(trits) + 1)
    ax2 = ax.twinx()

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    trit_colors = [colors[t] for t in trits]
    ax.bar(iterations, trits, color=trit_colors, alpha=0.6, label='Trit value')
    ax.set_ylabel('Trit value', color=COLORS['primary'])
    ax.set_yticks([0, 1, 2])

    ax2.semilogy(iterations, resolutions, 'o-', color=COLORS['danger'],
                 linewidth=1.5, markersize=4, label='Resolution')
    ax2.set_ylabel('Resolution (A)', color=COLORS['danger'])

    ax.set_xlabel('Iteration')
    ax.set_title('17 Trisection Iterations')


def _draw_sentropy_trajectory(ax, sentropy):
    sk = sentropy.get('sk', [])
    st = sentropy.get('st', [])
    se = sentropy.get('se', [])

    if not sk:
        ax.set_title('S-Entropy')
        return

    steps = range(len(sk))
    ax.fill_between(steps, 0, sk, alpha=0.3, color=COLORS['primary'], label='S_k')
    ax.fill_between(steps, sk, [a+b for a, b in zip(sk, st)], alpha=0.3,
                    color=COLORS['secondary'], label='S_t')
    ax.fill_between(steps, [a+b for a, b in zip(sk, st)],
                    [a+b+c for a, b, c in zip(sk, st, se)], alpha=0.3,
                    color=COLORS['tertiary'], label='S_e')

    ax.set_xlabel('Step')
    ax.set_ylabel('S-Entropy Component')
    ax.set_title('S-Entropy Conservation')
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def _draw_backaction_profile(ax, trisection):
    backactions = trisection.get('backactions', [])
    if not backactions:
        ax.set_title('Backaction')
        return

    iterations = range(1, len(backactions) + 1)
    mean_ba = trisection.get('mean_backaction', np.mean(backactions))

    ax.bar(iterations, backactions, color=COLORS['success'], alpha=0.7,
           edgecolor='white')
    ax.axhline(mean_ba, color=COLORS['danger'], linestyle='--', linewidth=1,
               label=f'Mean = {mean_ba:.2e}')
    ax.axhline(1e-3, color=COLORS['neutral'], linestyle=':', linewidth=1,
               label='QND limit')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Backaction delta')
    ax.set_title(f'Backaction: {mean_ba:.2e}')
    ax.legend(fontsize=6)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
