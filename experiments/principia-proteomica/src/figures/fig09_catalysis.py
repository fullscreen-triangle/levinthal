"""Figure 9: Enzyme Catalysis — d_C prediction and phase coherence."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    cat = results.get('catalysis', {})
    enzyme_pred = cat.get('enzyme_prediction', {})
    coherence = cat.get('catalytic_coherence', {})

    # Panel A: 3D enzyme landscape (dC, predicted, observed)
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    label_panel(ax_a, 'A', x=-0.05, y=1.02)
    _draw_3d_enzyme_landscape(ax_a, enzyme_pred)

    # Panel B: d_C bar chart by enzyme
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_dc_diagram(ax_b, enzyme_pred)

    # Panel C: 8-enzyme scatter
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_enzyme_scatter(ax_c, enzyme_pred)

    # Panel D: SOD1 catalytic coherence
    ax_d = fig.add_subplot(gs[0, 3])
    label_panel(ax_d, 'D')
    _draw_catalytic_coherence(ax_d, coherence)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_3d_enzyme_landscape(ax, enzyme_pred):
    """3D scatter of enzymes in (dC, predicted, observed) space."""
    enzymes = enzyme_pred.get('enzymes', [])
    if not enzymes:
        ax.set_title('Enzyme Landscape')
        return

    dc = [e['dC'] for e in enzymes]
    pred = [e['predicted'] for e in enzymes]
    obs = [e['observed'] for e in enzymes]
    names = [e['name'] for e in enzymes]

    ax.scatter(dc, pred, obs, s=60, c=COLORS['primary'], alpha=0.8,
               edgecolor='white')

    # Perfect prediction plane
    dc_range = np.linspace(min(dc) - 0.5, max(dc) + 0.5, 10)
    pred_range = np.linspace(min(pred) - 0.5, max(pred) + 0.5, 10)
    DC, PR = np.meshgrid(dc_range, pred_range)
    OBS = PR  # perfect prediction: observed = predicted
    ax.plot_surface(DC, PR, OBS, alpha=0.1, color=COLORS['success'])

    for d, p, o, name in zip(dc, pred, obs, names):
        ax.text(d, p, o + 0.2, name, fontsize=5, ha='center')

    ax.set_xlabel('d_C', fontsize=7)
    ax.set_ylabel('Predicted', fontsize=7)
    ax.set_zlabel('Observed', fontsize=7)
    ax.set_title('Enzyme Landscape')


def _draw_dc_diagram(ax, enzyme_pred):
    enzymes = enzyme_pred.get('enzymes', [])
    if not enzymes:
        ax.set_title('Categorical Distance')
        return

    names = [e['name'] for e in enzymes]
    dc_values = [e['dC'] for e in enzymes]

    colors = []
    for dc in dc_values:
        if dc == 1:
            colors.append(COLORS['success'])
        elif dc == 2:
            colors.append(COLORS['secondary'])
        elif dc == 3:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['danger'])

    bars = ax.barh(names, dc_values, color=colors, alpha=0.8, edgecolor='white')
    for bar, dc in zip(bars, dc_values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(dc), va='center', fontsize=7, fontweight='bold')

    ax.set_xlabel('d_C')
    ax.set_title('d_C by Enzyme')
    ax.invert_yaxis()


def _draw_enzyme_scatter(ax, enzyme_pred):
    enzymes = enzyme_pred.get('enzymes', [])
    if not enzymes:
        ax.set_title('Enzyme Prediction')
        return

    predicted = [e['predicted'] for e in enzymes]
    observed = [e['observed'] for e in enzymes]
    names = [e['name'] for e in enzymes]

    ax.scatter(predicted, observed, s=60, c=COLORS['primary'], alpha=0.8,
               edgecolor='white', linewidth=1, zorder=5)

    for p, o, name in zip(predicted, observed, names):
        ax.annotate(name, (p, o), textcoords='offset points',
                   xytext=(5, 5), fontsize=5.5)

    ax.plot([3, 10], [3, 10], '--', color=COLORS['neutral'], linewidth=1,
            label='Perfect')
    ax.fill_between([3, 10], [2, 9], [4, 11], alpha=0.1,
                    color=COLORS['success'], label='+/-1 log')

    mae = enzyme_pred.get('mae', 0)
    ax.set_xlabel('Predicted log10(kcat/Km)')
    ax.set_ylabel('Observed log10(kcat/Km)')
    ax.set_title(f'MAE = {mae:.2f}')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def _draw_catalytic_coherence(ax, coherence):
    r_series = coherence.get('r_series', [])
    if not r_series:
        ax.set_title('Catalytic Coherence')
        return

    ax.plot(range(len(r_series)), r_series, '-', color=COLORS['primary'],
            linewidth=1, alpha=0.8)
    ax.axhline(0.99, color=COLORS['success'], linestyle='--', linewidth=0.8,
               label='r = 0.99')
    ax.axhline(0.9, color=COLORS['secondary'], linestyle='--', linewidth=0.8,
               label='r = 0.90')

    mean_r = coherence.get('mean_steady_state_r', 0)
    ax.set_xlabel('Time step')
    ax.set_ylabel('r')
    ax.set_title(f'SOD1 Coherence (r = {mean_r:.3f})')
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
