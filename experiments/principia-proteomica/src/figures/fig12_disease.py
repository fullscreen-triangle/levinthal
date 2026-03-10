"""Figure 12: Disease / ALS — Variant coherence and survival correlation."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    disease = results.get('disease', {})
    variants = disease.get('variants', {}).get('variants', [])
    survival = disease.get('survival_correlation', {})
    chaperone = disease.get('chaperone_rescue', {})

    # Panel A: 3D variant landscape (r, survival, perturbation)
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    label_panel(ax_a, 'A', x=-0.05, y=1.02)
    _draw_3d_variant_landscape(ax_a, variants, survival)

    # Panel B: Variant r bar chart
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_variant_bars(ax_b, variants)

    # Panel C: Survival correlation
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_survival_fit(ax_c, survival, variants)

    # Panel D: Chaperone rescue
    ax_d = fig.add_subplot(gs[0, 3])
    label_panel(ax_d, 'D')
    _draw_chaperone_rescue(ax_d, chaperone)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_3d_variant_landscape(ax, variants, survival):
    """3D scatter: variant r, survival years, target r."""
    if not variants:
        ax.set_title('Variant Landscape')
        return

    mutant_variants = [v for v in variants if v.get('survival_years') is not None]
    if not mutant_variants:
        ax.set_title('Variant Landscape')
        return

    r_vals = [v['measured_r'] for v in mutant_variants]
    targets = [v['target_r'] for v in mutant_variants]
    survival_yrs = [v['survival_years'] for v in mutant_variants]
    names = [v['name'] for v in mutant_variants]

    ax.scatter(r_vals, targets, survival_yrs, s=60, c=COLORS['primary'],
               alpha=0.8, edgecolor='white')

    for r, tgt, s, name in zip(r_vals, targets, survival_yrs, names):
        ax.text(r, tgt, s + 0.5, name, fontsize=5.5, ha='center')

    ax.set_xlabel('Measured r', fontsize=7)
    ax.set_ylabel('Target r', fontsize=7)
    ax.set_zlabel('Survival (yr)', fontsize=7)
    ax.set_title('Variant Landscape')


def _draw_variant_bars(ax, variants):
    if not variants:
        ax.set_title('Variant Coherence')
        return

    names = [v['name'] for v in variants]
    r_values = [v['measured_r'] for v in variants]
    targets = [v['target_r'] for v in variants]

    colors = []
    for r in r_values:
        if r > 0.7:
            colors.append(COLORS['success'])
        elif r > 0.5:
            colors.append(COLORS['secondary'])
        else:
            colors.append(COLORS['danger'])

    x = np.arange(len(names))
    bars = ax.bar(x, r_values, color=colors, alpha=0.8, edgecolor='white',
                  label='Measured')
    ax.scatter(x, targets, marker='_', s=100, color=COLORS['dark'],
               linewidth=2, zorder=5, label='Target')

    for bar, r in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{r:.2f}', ha='center', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('r')
    ax.set_title('SOD1 Variant Coherence')
    ax.axhline(0.5, color=COLORS['neutral'], linestyle='--', linewidth=0.8,
               alpha=0.5)
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.1)


def _draw_survival_fit(ax, survival, variants):
    if not survival:
        ax.set_title('Survival Correlation')
        return

    r_values = survival.get('r_values', [])
    observed = survival.get('observed_survival', [])
    correlation = survival.get('correlation', 0)

    if not r_values:
        return

    mutant_variants = [v for v in variants if v.get('survival_years') is not None]
    names = [v['name'] for v in mutant_variants]

    ax.scatter(r_values, observed, s=60, c=COLORS['primary'], alpha=0.8,
               edgecolor='white', zorder=5)

    for r, s, name in zip(r_values, observed, names):
        ax.annotate(name, (r, s), textcoords='offset points',
                   xytext=(5, 5), fontsize=6)

    if len(r_values) > 1:
        r_arr = np.array(r_values)
        s_arr = np.array(observed)
        coeffs = np.polyfit(r_arr, np.log(s_arr), 1)
        r_fit = np.linspace(min(r_arr) - 0.05, max(r_arr) + 0.05, 50)
        s_fit = np.exp(np.polyval(coeffs, r_fit))
        ax.plot(r_fit, s_fit, '--', color=COLORS['danger'], linewidth=1.5)

    ax.set_xlabel('r')
    ax.set_ylabel('Survival (years)')
    ax.set_title(f'Survival (rho = {correlation:.3f})')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)


def _draw_chaperone_rescue(ax, chaperone):
    if not chaperone:
        ax.set_title('Chaperone Rescue')
        return

    r_no = chaperone.get('r_series_no_chaperone', [])
    r_chap = chaperone.get('r_series_chaperone', [])

    if r_no:
        ax.plot(range(len(r_no)), r_no, '-', color=COLORS['danger'],
                linewidth=1, alpha=0.7, label='No chaperone')
    if r_chap:
        ax.plot(range(len(r_chap)), r_chap, '-', color=COLORS['success'],
                linewidth=1, alpha=0.7, label='With chaperone')

    ax.axhline(0.5, color=COLORS['neutral'], linestyle='--', linewidth=0.8)

    mutation = chaperone.get('mutation', 'A4V')
    delta_r = chaperone.get('r_improvement', 0)

    ax.set_xlabel('Time step')
    ax.set_ylabel('r')
    ax.set_title(f'{mutation} Rescue (dr = {delta_r:.3f})')
    ax.legend(fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
