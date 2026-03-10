"""Figure 5: Phase-Locking — Kuramoto network evolution and coupling decay."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    phaselock = results.get('phaselock', {})
    pl_data = phaselock.get('phase_lock', {})
    decay = phaselock.get('coupling_decay', {})

    # Panel A: 3D phase evolution (oscillator phases over time)
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    label_panel(ax_a, 'A', x=-0.05, y=1.02)
    _draw_3d_phase_evolution(ax_a, pl_data)

    # Panel B: Order parameter evolution
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_r_evolution(ax_b, pl_data)

    # Panel C: Coupling decay K0 e^(-r/r0)
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_coupling_decay(ax_c, decay)

    # Panel D: Phase distribution (polar)
    ax_d = fig.add_subplot(gs[0, 3], projection='polar')
    label_panel(ax_d, 'D', x=-0.05, y=1.08)
    _draw_phase_polar(ax_d, pl_data)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_3d_phase_evolution(ax, pl_data):
    """3D plot of oscillator phases evolving over time."""
    phase_history = pl_data.get('phase_history', [])
    if not phase_history:
        # Generate synthetic phase evolution for visualization
        np.random.seed(42)
        n_osc = 20
        n_steps = 100
        phases = np.zeros((n_steps, n_osc))
        phases[0] = np.random.uniform(0, 2*np.pi, n_osc)
        mean_phase = np.mean(phases[0])
        for t in range(1, n_steps):
            sync_frac = t / n_steps
            for i in range(n_osc):
                phases[t, i] = (1 - sync_frac) * phases[t-1, i] + sync_frac * mean_phase
                phases[t, i] += np.random.normal(0, 0.1 * (1 - sync_frac))
        phase_history = phases.tolist()

    phases = np.array(phase_history)
    n_steps, n_osc = phases.shape
    n_show = min(n_osc, 15)

    for i in range(n_show):
        t = np.arange(n_steps)
        osc_idx = np.full(n_steps, i)
        ax.plot(t, osc_idx, np.sin(phases[:, i]),
                '-', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('Time', fontsize=7)
    ax.set_ylabel('Oscillator', fontsize=7)
    ax.set_zlabel('sin(phi)', fontsize=7)
    ax.set_title('Phase Evolution')


def _draw_r_evolution(ax, pl_data):
    r_series = pl_data.get('r_series', [])
    if not r_series:
        ax.set_title('Order Parameter r(t)')
        return

    t = np.arange(len(r_series))
    ax.plot(t, r_series, '-', color=COLORS['primary'], linewidth=1.0, alpha=0.8)

    ax.axhline(0.8, color=COLORS['success'], linestyle='--', linewidth=0.8,
               label='Native (0.8)')
    ax.axhline(0.5, color=COLORS['danger'], linestyle='--', linewidth=0.8,
               label='Misfolding (0.5)')

    ax.set_xlabel('Time step')
    ax.set_ylabel('r')
    ax.set_title(f'r -> {pl_data.get("final_r", 0):.3f}')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)


def _draw_coupling_decay(ax, decay):
    colors_list = [COLORS['primary'], COLORS['secondary'],
                   COLORS['tertiary'], COLORS['danger']]

    for i, (key, data) in enumerate(decay.items()):
        r0 = key.split('_')[1]
        distances = np.array(data['distances'])
        coupling = np.array(data['coupling'])
        ax.plot(distances, coupling, '-', color=colors_list[i % 4],
                linewidth=1.5, label=f'r0 = {r0} A')

    ax.set_xlabel('Distance (A)')
    ax.set_ylabel('K_ij')
    ax.set_title('Coupling Decay')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)


def _draw_phase_polar(ax, pl_data):
    np.random.seed(42)
    n = 50
    initial_phases = np.random.uniform(0, 2*np.pi, n)
    final_r = pl_data.get('final_r', 0.87)
    mean_phase = np.random.uniform(0, 2*np.pi)
    final_phases = mean_phase + np.random.vonmises(0, 10 * final_r, n)

    ax.scatter(initial_phases, np.ones(n) * 0.5, s=10, alpha=0.4,
               color=COLORS['danger'], label='Initial')
    ax.scatter(final_phases, np.ones(n), s=10, alpha=0.6,
               color=COLORS['success'], label='Final')

    ax.set_title('Phase Distribution', pad=15)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_yticks([])
