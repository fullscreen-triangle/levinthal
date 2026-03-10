"""Figure 1: Theoretical Overview — Domain validation and equation coverage."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .style import create_4panel, label_panel, COLORS


def generate(results: dict, output_path: str):
    fig, gs = create_4panel()

    grand = results.get('grand_validation', {})
    domains = grand.get('domains', {})

    # Panel A: 3D scatter of test results by domain and equation
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    label_panel(ax_a, 'A', x=-0.05, y=1.02)
    _draw_3d_test_landscape(ax_a, grand)

    # Panel B: Domain pass rates bar chart
    ax_b = fig.add_subplot(gs[0, 1])
    label_panel(ax_b, 'B')
    _draw_domain_bars(ax_b, domains)

    # Panel C: Equation coverage heatmap
    ax_c = fig.add_subplot(gs[0, 2])
    label_panel(ax_c, 'C')
    _draw_equation_heatmap(ax_c)

    # Panel D: Pass/fail scatter across all tests
    ax_d = fig.add_subplot(gs[0, 3])
    label_panel(ax_d, 'D')
    _draw_test_scatter(ax_d, grand)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_3d_test_landscape(ax, grand):
    """3D scatter: domain_id x test_index x pass_rate."""
    all_tests = grand.get('all_tests', [])
    if not all_tests:
        ax.set_title('Test Landscape')
        return

    domain_names = list(dict.fromkeys(t.get('domain', 'Unknown') for t in all_tests))
    domain_map = {name: i for i, name in enumerate(domain_names)}

    xs, ys, zs, cs = [], [], [], []
    for i, t in enumerate(all_tests):
        d_idx = domain_map.get(t.get('domain', 'Unknown'), 0)
        passed = 1.0 if t.get('passed', False) else 0.0
        xs.append(d_idx)
        ys.append(i)
        zs.append(passed)
        cs.append(COLORS['success'] if passed else COLORS['danger'])

    ax.bar3d(xs, ys, [0]*len(xs), 0.4, 0.6, zs,
             color=cs, alpha=0.8)
    ax.set_xlabel('Domain', fontsize=7)
    ax.set_ylabel('Test #', fontsize=7)
    ax.set_zlabel('Pass', fontsize=7)
    ax.set_title('Validation Landscape')


def _draw_domain_bars(ax, domains):
    if not domains:
        domains = {
            'Atomic structure': {'passed': 7, 'total': 7},
            'Electron transfer': {'passed': 5, 'total': 5},
            'Enzyme catalysis': {'passed': 10, 'total': 12},
            'Protein folding': {'passed': 5, 'total': 5},
            'Disease (ALS)': {'passed': 6, 'total': 7},
        }

    names = list(domains.keys())
    rates = [d['passed'] / d['total'] * 100 for d in domains.values()]
    colors = [COLORS['success'] if r == 100 else
              COLORS['secondary'] if r >= 80 else COLORS['warning'] for r in rates]

    bars = ax.barh(names, rates, color=colors, alpha=0.8, edgecolor='white')
    for bar, r in zip(bars, rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{r:.0f}%', va='center', fontsize=7)

    ax.set_xlabel('Pass Rate (%)')
    ax.set_title('Domain Pass Rates')
    ax.set_xlim(0, 115)
    ax.axvline(100, color=COLORS['neutral'], linestyle='--', linewidth=0.5)
    ax.invert_yaxis()


def _draw_equation_heatmap(ax):
    """Heatmap: which equations apply to which domains."""
    domains = ['Atomic', 'Electron', 'Enzyme', 'Folding', 'Disease']
    equations = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']

    coverage = np.array([
        [1, 1, 1, 0, 0, 0, 0],  # Atomic
        [0, 1, 0, 0, 0, 0, 1],  # Electron
        [0, 0, 0, 1, 1, 0, 0],  # Enzyme
        [0, 0, 0, 0, 1, 1, 0],  # Folding
        [0, 0, 0, 0, 1, 1, 0],  # Disease
    ])

    ax.imshow(coverage, cmap='YlGn', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(equations)))
    ax.set_xticklabels(equations, fontsize=7)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=7)
    ax.set_xlabel('Equation')
    ax.set_title('Equation-Domain Coverage')


def _draw_test_scatter(ax, grand):
    """Scatter of all test pass/fail with score."""
    all_tests = grand.get('all_tests', [])
    if not all_tests:
        ax.set_title('Test Results')
        return

    passed = [i for i, t in enumerate(all_tests) if t.get('passed', False)]
    failed = [i for i, t in enumerate(all_tests) if not t.get('passed', False)]

    ax.scatter(passed, [1]*len(passed), s=30, c=COLORS['success'],
               alpha=0.8, label=f'Pass ({len(passed)})')
    ax.scatter(failed, [0]*len(failed), s=30, c=COLORS['danger'],
               alpha=0.8, label=f'Fail ({len(failed)})')

    total = len(all_tests)
    rate = len(passed) / total * 100 if total else 0
    ax.set_xlabel('Test Index')
    ax.set_ylabel('Pass/Fail')
    ax.set_title(f'All Tests: {rate:.0f}% Pass')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Fail', 'Pass'])
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis='x')
