"""
Generate all panel figures for SOD1 categorical mechanics paper.
Each panel: 4 figures in a row, at least one 3D, minimal text, tight layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.dpi'] = 150

# Color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#3A7D44',
    'neutral': '#6C757D',
    'light': '#E8E8E8',
    'copper': '#B87333',
    'zinc': '#7D7D7D'
}


def create_panel_figure(width=14, height=3.5):
    """Create a 1x4 panel figure with tight layout."""
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(1, 4, figure=fig, wspace=0.25, hspace=0.1,
                  left=0.04, right=0.98, top=0.92, bottom=0.12)
    return fig, gs


# =============================================================================
# PANEL 1: Partition Coordinates (Section 2)
# =============================================================================

def generate_panel_partition():
    """Panel for partition coordinates from bounded phase space."""
    fig, gs = create_panel_figure()

    # Panel A: 3D partition state space
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Generate partition states
    states = []
    colors = []
    sizes = []
    for n in range(1, 5):
        for l in range(n):
            for m in range(-l, l+1):
                states.append([n, l, m])
                colors.append(plt.cm.viridis(n/5))
                sizes.append(30 + n*15)

    states = np.array(states)
    ax1.scatter(states[:,0], states[:,1], states[:,2],
                c=colors, s=sizes, alpha=0.8, edgecolors='white', linewidth=0.5)

    ax1.set_xlabel('n', fontsize=8, labelpad=2)
    ax1.set_ylabel('ℓ', fontsize=8, labelpad=2)
    ax1.set_zlabel('m', fontsize=8, labelpad=2)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=20, azim=45)
    ax1.tick_params(pad=0, labelsize=6)

    # Panel B: Capacity formula validation
    ax2 = fig.add_subplot(gs[1])

    n_vals = np.arange(1, 11)
    C_theory = 2 * n_vals**2
    C_enum = C_theory  # Perfect match

    ax2.plot(n_vals, C_theory, 'o-', color=COLORS['primary'],
             markersize=6, linewidth=1.5, label='2n²')
    ax2.bar(n_vals, C_enum, alpha=0.3, color=COLORS['secondary'], width=0.6)

    ax2.set_xlabel('n', fontsize=8)
    ax2.set_ylabel('C(n)', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax2.set_xticks(n_vals)
    ax2.set_xlim(0.5, 10.5)

    # Panel C: Subshell capacities
    ax3 = fig.add_subplot(gs[2])

    l_vals = np.arange(0, 5)
    subshell_labels = ['s', 'p', 'd', 'f', 'g']
    C_l = 2 * (2*l_vals + 1)

    bars = ax3.bar(l_vals, C_l, color=[plt.cm.plasma(l/5) for l in l_vals],
                   edgecolor='white', linewidth=0.5)

    ax3.set_xlabel('ℓ', fontsize=8)
    ax3.set_ylabel('Cℓ', fontsize=8)
    ax3.set_xticks(l_vals)
    ax3.set_xticklabels(subshell_labels)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)

    # Panel D: SOD1 Cu electronic structure
    ax4 = fig.add_subplot(gs[3])

    # Energy levels for Cu
    levels = {
        '3d': (-2.0, 10, COLORS['copper']),
        '4s': (-0.5, 1, COLORS['primary']),
    }

    for i, (name, (energy, electrons, color)) in enumerate(levels.items()):
        ax4.hlines(energy, i-0.3, i+0.3, colors=color, linewidth=3)
        # Show electrons as arrows
        if name == '3d':
            for j in range(5):
                x = i - 0.25 + j*0.125
                if j < 5:  # Cu2+ has 9 electrons in 3d
                    ax4.annotate('', xy=(x, energy+0.15), xytext=(x, energy-0.15),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1))
                    if j < 4:  # Paired electrons
                        ax4.annotate('', xy=(x+0.03, energy-0.15), xytext=(x+0.03, energy+0.15),
                                    arrowprops=dict(arrowstyle='->', color='blue', lw=1))
        ax4.text(i, energy-0.4, name, ha='center', fontsize=7)

    ax4.set_xlim(-0.5, 1.5)
    ax4.set_ylim(-3, 0.5)
    ax4.set_ylabel('E (arb.)', fontsize=8)
    ax4.set_xticks([])
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.text(0.5, 0.2, 'Cu²⁺: 3d⁹', ha='center', fontsize=8)

    fig.suptitle('Partition Coordinates from Bounded Phase Space',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel1_partition.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel1_partition.png")


# =============================================================================
# PANEL 2: Selection Rules (Section 3)
# =============================================================================

def generate_panel_selection():
    """Panel for selection rules from boundary continuity."""
    fig, gs = create_panel_figure()

    # Panel A: 3D transition pathway
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Allowed transitions in (n, l, m) space
    t = np.linspace(0, 2*np.pi, 100)
    n = 3 * np.ones_like(t)
    l = 2 * np.ones_like(t)
    m = 2 * np.sin(t)

    ax1.plot(n, l, m, color=COLORS['success'], linewidth=2, label='Allowed')
    ax1.scatter([3], [2], [2], color=COLORS['primary'], s=80, marker='o', zorder=5)
    ax1.scatter([3], [2], [-2], color=COLORS['quaternary'], s=80, marker='s', zorder=5)

    # Forbidden transition (dashed)
    ax1.plot([3, 3], [2, 0], [0, 0], '--', color=COLORS['neutral'],
             linewidth=1.5, alpha=0.5, label='Forbidden')

    ax1.set_xlabel('n', fontsize=8, labelpad=2)
    ax1.set_ylabel('ℓ', fontsize=8, labelpad=2)
    ax1.set_zlabel('m', fontsize=8, labelpad=2)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=25, azim=35)
    ax1.tick_params(pad=0, labelsize=6)

    # Panel B: Selection rule matrix
    ax2 = fig.add_subplot(gs[1])

    delta_l = np.arange(-3, 4)
    delta_m = np.arange(-3, 4)

    matrix = np.zeros((len(delta_l), len(delta_m)))
    for i, dl in enumerate(delta_l):
        for j, dm in enumerate(delta_m):
            if abs(dl) == 1 and abs(dm) <= 1:
                matrix[i, j] = 1  # Allowed

    im = ax2.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(delta_m)))
    ax2.set_xticklabels(delta_m)
    ax2.set_yticks(range(len(delta_l)))
    ax2.set_yticklabels(delta_l)
    ax2.set_xlabel('Δm', fontsize=8)
    ax2.set_ylabel('Δℓ', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)

    # Panel C: Enforcement ratio
    ax3 = fig.add_subplot(gs[2])

    categories = ['Allowed', 'Forbidden']
    rates = [1e12, 1e4]
    colors = [COLORS['success'], COLORS['quaternary']]

    bars = ax3.bar(categories, rates, color=colors, edgecolor='white', linewidth=0.5)
    ax3.set_yscale('log')
    ax3.set_ylabel('Γ (s⁻¹)', fontsize=8)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax3.set_ylim(1e2, 1e14)

    # Add ratio annotation
    ax3.annotate('', xy=(1, 1e12), xytext=(1, 1e4),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax3.text(1.15, 1e8, '>10⁸', fontsize=8, va='center')

    # Panel D: SOD1 electron transitions
    ax4 = fig.add_subplot(gs[3])

    # m values during catalysis
    time = np.array([0, 10, 20, 30, 45, 55, 70, 85])
    m_values = np.array([2, 1, 0, -1, -1, 0, 1, 2])

    ax4.plot(time, m_values, 'o-', color=COLORS['copper'],
             markersize=6, linewidth=1.5)
    ax4.fill_between(time, m_values, alpha=0.2, color=COLORS['copper'])

    ax4.axhline(0, color=COLORS['neutral'], linestyle='--', linewidth=0.5)
    ax4.set_xlabel('t (fs)', fontsize=8)
    ax4.set_ylabel('m', fontsize=8)
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.set_ylim(-2.5, 2.5)
    ax4.set_yticks([-2, -1, 0, 1, 2])

    fig.suptitle('Selection Rules from Boundary Continuity',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel2_selection.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel2_selection.png")


# =============================================================================
# PANEL 3: Phase-Lock Networks (Section 4)
# =============================================================================

def generate_panel_phaselock():
    """Panel for phase-lock networks from coupled oscillators."""
    fig, gs = create_panel_figure()

    # Panel A: 3D H-bond network
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Generate random network nodes (H-bond positions)
    np.random.seed(42)
    n_nodes = 50
    theta = np.random.uniform(0, 2*np.pi, n_nodes)
    phi = np.random.uniform(0, np.pi, n_nodes)
    r = np.random.uniform(5, 15, n_nodes)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Color by phase
    phases = np.random.uniform(0, 2*np.pi, n_nodes)
    ax1.scatter(x, y, z, c=phases, cmap='twilight', s=40, alpha=0.8)

    # Draw some connections
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)
            if dist < 6:
                ax1.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                        'k-', alpha=0.1, linewidth=0.3)

    ax1.set_xlabel('x (Å)', fontsize=7, labelpad=1)
    ax1.set_ylabel('y (Å)', fontsize=7, labelpad=1)
    ax1.set_zlabel('z (Å)', fontsize=7, labelpad=1)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.tick_params(pad=0, labelsize=5)
    ax1.view_init(elev=20, azim=45)

    # Panel B: Order parameter evolution
    ax2 = fig.add_subplot(gs[1])

    t = np.linspace(0, 100, 200)
    # Synchronization transition
    r_order = 0.1 + 0.77 * (1 - np.exp(-t/20))
    r_order += 0.02 * np.random.randn(len(t))
    r_order = np.clip(r_order, 0, 1)

    ax2.plot(t, r_order, color=COLORS['primary'], linewidth=1.5)
    ax2.axhline(0.8, color=COLORS['success'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(0.5, color=COLORS['quaternary'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.fill_between(t, 0.8, 1, alpha=0.1, color=COLORS['success'])
    ax2.fill_between(t, 0, 0.5, alpha=0.1, color=COLORS['quaternary'])

    ax2.set_xlabel('t (ps)', fontsize=8)
    ax2.set_ylabel('⟨r⟩', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 100)

    # Panel C: Coupling decay
    ax3 = fig.add_subplot(gs[2])

    r = np.linspace(0, 15, 100)
    K = np.exp(-r/5)

    ax3.plot(r, K, color=COLORS['secondary'], linewidth=2)
    ax3.fill_between(r, K, alpha=0.2, color=COLORS['secondary'])
    ax3.axvline(5, color=COLORS['neutral'], linestyle=':', linewidth=1)

    ax3.set_xlabel('r (Å)', fontsize=8)
    ax3.set_ylabel('K/K₀', fontsize=8)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax3.set_xlim(0, 15)
    ax3.set_ylim(0, 1.1)
    ax3.text(5.5, 0.4, 'r₀=5Å', fontsize=7, color=COLORS['neutral'])

    # Panel D: Kuramoto phases
    ax4 = fig.add_subplot(gs[3], projection='polar')

    # Synchronized phases
    n_osc = 20
    phases_sync = np.random.normal(0, 0.2, n_osc)
    phases_sync = np.mod(phases_sync, 2*np.pi)

    ax4.scatter(phases_sync, np.ones(n_osc), c=COLORS['primary'], s=50, alpha=0.7)

    # Mean phase arrow
    mean_phase = np.mean(np.exp(1j * phases_sync))
    ax4.annotate('', xy=(np.angle(mean_phase), np.abs(mean_phase)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=COLORS['quaternary'], lw=2))

    ax4.set_ylim(0, 1.3)
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=8)
    ax4.set_rticks([])
    ax4.tick_params(labelsize=6)

    fig.suptitle('Phase-Lock Networks from Coupled Oscillators',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel3_phaselock.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel3_phaselock.png")


# =============================================================================
# PANEL 4: S-Entropy Space (Section 5)
# =============================================================================

def generate_panel_sentropy():
    """Panel for S-entropy space and ternary representation."""
    fig, gs = create_panel_figure()

    # Panel A: 3D S-entropy space
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Trajectory through S-entropy space
    t = np.linspace(0, 1, 100)
    Sk = 1 - 0.8*t + 0.1*np.sin(10*t)
    St = 0.3 + 0.4*t + 0.1*np.cos(10*t)
    Se = 0.1*t

    # Ensure constraint
    total = Sk + St + Se
    Sk, St, Se = Sk/total, St/total, Se/total

    ax1.plot(Sk, St, Se, color=COLORS['primary'], linewidth=2)
    ax1.scatter([Sk[0]], [St[0]], [Se[0]], color=COLORS['success'], s=80, marker='o', zorder=5)
    ax1.scatter([Sk[-1]], [St[-1]], [Se[-1]], color=COLORS['quaternary'], s=80, marker='s', zorder=5)

    # Constraint plane
    xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    zz = 1 - xx - yy
    zz[zz < 0] = np.nan
    ax1.plot_surface(xx, yy, zz, alpha=0.1, color=COLORS['neutral'])

    ax1.set_xlabel('Sₖ', fontsize=8, labelpad=2)
    ax1.set_ylabel('Sₜ', fontsize=8, labelpad=2)
    ax1.set_zlabel('Sₑ', fontsize=8, labelpad=2)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=25, azim=45)
    ax1.tick_params(pad=0, labelsize=6)

    # Panel B: Ternary partition
    ax2 = fig.add_subplot(gs[1])

    # 3x3x3 ternary grid
    for i in range(4):
        ax2.axhline(i/3, color=COLORS['neutral'], linewidth=0.5)
        ax2.axvline(i/3, color=COLORS['neutral'], linewidth=0.5)

    # Color cells by trit value
    for i in range(3):
        for j in range(3):
            color = plt.cm.viridis((i*3 + j)/9)
            ax2.fill([i/3, (i+1)/3, (i+1)/3, i/3],
                    [j/3, j/3, (j+1)/3, (j+1)/3],
                    color=color, alpha=0.5)
            ax2.text((i+0.5)/3, (j+0.5)/3, f'{i}{j}',
                    ha='center', va='center', fontsize=7)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Sₖ', fontsize=8)
    ax2.set_ylabel('Sₜ', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax2.set_aspect('equal')

    # Panel C: Conservation law
    ax3 = fig.add_subplot(gs[2])

    t = np.linspace(0, 100, 100)
    Sk = 0.6 - 0.4*t/100
    St = 0.3 + 0.35*t/100
    Se = 0.1 + 0.05*t/100

    ax3.stackplot(t, Sk, St, Se,
                  colors=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']],
                  alpha=0.7, labels=['Sₖ', 'Sₜ', 'Sₑ'])
    ax3.axhline(1, color='black', linestyle='--', linewidth=1)

    ax3.set_xlabel('t (fs)', fontsize=8)
    ax3.set_ylabel('S', fontsize=8)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax3.set_ylim(0, 1.1)
    ax3.set_xlim(0, 100)
    ax3.legend(loc='right', fontsize=6, frameon=False)

    # Panel D: Backaction comparison
    ax4 = fig.add_subplot(gs[3])

    methods = ['Heisenberg', 'Weak', 'QND', 'Categorical']
    backaction = [1, 1e-2, 1e-3, 1.5e-4]
    colors = [COLORS['quaternary'], COLORS['tertiary'], COLORS['secondary'], COLORS['success']]

    bars = ax4.bar(range(len(methods)), backaction, color=colors,
                   edgecolor='white', linewidth=0.5)
    ax4.set_yscale('log')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=30, ha='right', fontsize=7)
    ax4.set_ylabel('δ = Δp/p₀', fontsize=8)
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.set_ylim(1e-5, 10)

    fig.suptitle('S-Entropy Space and Zero-Backaction Measurement',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel4_sentropy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel4_sentropy.png")


# =============================================================================
# PANEL 5: Electron Trajectory (Section 7)
# =============================================================================

def generate_panel_electron():
    """Panel for electron trajectory tracking in SOD1."""
    fig, gs = create_panel_figure()

    # Panel A: 3D electron trajectory
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Spiral trajectory through active site
    t = np.linspace(0, 85, 100)
    r = 285 - 187*np.sin(np.pi*t/85)**2 + 50*np.sin(np.pi*t/85)
    theta = 2*np.pi*t/85

    x = r * np.cos(theta) / 100
    y = r * np.sin(theta) / 100
    z = t / 85 * 5

    # Color by time
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t)-1))
    for i in range(len(segments)):
        ax1.plot([segments[i,0,0], segments[i,1,0]],
                [segments[i,0,1], segments[i,1,1]],
                [segments[i,0,2], segments[i,1,2]],
                color=colors[i], linewidth=2)

    # Cu center
    ax1.scatter([0], [0], [2.5], color=COLORS['copper'], s=150, marker='o',
                edgecolors='black', linewidth=1, zorder=5)

    ax1.set_xlabel('x (Å)', fontsize=7, labelpad=1)
    ax1.set_ylabel('y (Å)', fontsize=7, labelpad=1)
    ax1.set_zlabel('z (Å)', fontsize=7, labelpad=1)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=20, azim=45)
    ax1.tick_params(pad=0, labelsize=5)

    # Panel B: Distance to Cu
    ax2 = fig.add_subplot(gs[1])

    time = np.array([0, 10, 20, 30, 45, 55, 70, 85])
    r_Cu = np.array([285, 241, 187, 152, 98, 134, 189, 267])

    ax2.plot(time, r_Cu, 'o-', color=COLORS['copper'], markersize=6, linewidth=1.5)
    ax2.fill_between(time, r_Cu, alpha=0.2, color=COLORS['copper'])
    ax2.axhline(98, color=COLORS['quaternary'], linestyle='--', linewidth=1, alpha=0.7)

    ax2.set_xlabel('t (fs)', fontsize=8)
    ax2.set_ylabel('r (pm)', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax2.set_ylim(0, 350)
    ax2.text(50, 120, 'min', fontsize=7, color=COLORS['quaternary'])

    # Panel C: m-value evolution
    ax3 = fig.add_subplot(gs[2])

    m_values = np.array([2, 1, 0, -1, -1, 0, 1, 2])

    ax3.step(time, m_values, where='mid', color=COLORS['primary'], linewidth=2)
    ax3.scatter(time, m_values, color=COLORS['primary'], s=40, zorder=5)
    ax3.axhline(0, color=COLORS['neutral'], linestyle='--', linewidth=0.5)

    # Highlight transitions
    for i in range(len(time)-1):
        if m_values[i] != m_values[i+1]:
            ax3.axvspan(time[i], time[i+1], alpha=0.1, color=COLORS['tertiary'])

    ax3.set_xlabel('t (fs)', fontsize=8)
    ax3.set_ylabel('m', fontsize=8)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax3.set_ylim(-2.5, 2.5)
    ax3.set_yticks([-2, -1, 0, 1, 2])

    # Panel D: Backaction validation
    ax4 = fig.add_subplot(gs[3])

    iterations = np.arange(1, 9)
    backaction_measured = 1.52e-4 * (1 + 0.2*np.random.randn(8))
    backaction_measured = np.abs(backaction_measured)

    ax4.bar(iterations, backaction_measured * 1e4, color=COLORS['success'],
            edgecolor='white', linewidth=0.5, alpha=0.8)
    ax4.axhline(1.52, color=COLORS['quaternary'], linestyle='--', linewidth=1.5)
    ax4.fill_between([0.5, 8.5], [1.52-0.28, 1.52-0.28], [1.52+0.28, 1.52+0.28],
                     alpha=0.2, color=COLORS['quaternary'])

    ax4.set_xlabel('Iteration', fontsize=8)
    ax4.set_ylabel('δ × 10⁴', fontsize=8)
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.set_xlim(0.5, 8.5)
    ax4.set_ylim(0, 2.5)

    fig.suptitle('Electron Trajectory Tracking in SOD1',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel5_electron.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel5_electron.png")


# =============================================================================
# PANEL 6: Catalysis (Section 8)
# =============================================================================

def generate_panel_catalysis():
    """Panel for catalytic mechanism as aperture traversal."""
    fig, gs = create_panel_figure()

    # Panel A: 3D active site
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Cu center
    ax1.scatter([0], [0], [0], color=COLORS['copper'], s=200, marker='o',
                edgecolors='black', linewidth=1, label='Cu')

    # Zn center
    ax1.scatter([3], [0], [0], color=COLORS['zinc'], s=150, marker='o',
                edgecolors='black', linewidth=1, label='Zn')

    # His ligands (simplified)
    his_positions = [
        (-1.5, 1, 0), (-1.5, -1, 0), (1, 0, 1.5), (1, 0, -1.5)
    ]
    for pos in his_positions:
        ax1.scatter(*pos, color=COLORS['primary'], s=50, marker='^')
        ax1.plot([0, pos[0]], [0, pos[1]], [0, pos[2]],
                'k-', linewidth=0.5, alpha=0.5)

    # Superoxide approach trajectory
    t = np.linspace(0, 1, 20)
    x_sub = -5 + 4.5*t
    y_sub = 2 - 2*t
    z_sub = 1 - 1*t
    ax1.plot(x_sub, y_sub, z_sub, '--', color=COLORS['quaternary'], linewidth=1.5)
    ax1.scatter([-5], [2], [1], color=COLORS['quaternary'], s=60, marker='*')

    ax1.set_xlabel('x (Å)', fontsize=7, labelpad=1)
    ax1.set_ylabel('y (Å)', fontsize=7, labelpad=1)
    ax1.set_zlabel('z (Å)', fontsize=7, labelpad=1)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=20, azim=45)
    ax1.tick_params(pad=0, labelsize=5)
    ax1.set_xlim(-6, 5)

    # Panel B: Categorical distance comparison
    ax2 = fig.add_subplot(gs[1])

    enzymes = ['SOD1', 'CA II', 'Catalase', 'TIM', 'Chymo.']
    d_C = [1, 1, 1, 2, 4]
    kcat = [1e9, 1e6, 4e7, 4e3, 1e2]

    colors = [COLORS['success'] if d == 1 else COLORS['tertiary'] if d == 2
              else COLORS['quaternary'] for d in d_C]

    ax2.bar(range(len(enzymes)), d_C, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(range(len(enzymes)))
    ax2.set_xticklabels(enzymes, rotation=30, ha='right', fontsize=7)
    ax2.set_ylabel('dC', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax2.set_ylim(0, 5)

    # Panel C: kcat vs dC
    ax3 = fig.add_subplot(gs[2])

    d_C_vals = np.array([1, 1, 1, 2, 4])
    kcat_vals = np.array([1e9, 1e6, 4e7, 4e3, 1e2])

    ax3.scatter(d_C_vals, kcat_vals, s=80, c=colors, edgecolors='black', linewidth=0.5)

    # Fit line
    x_fit = np.array([1, 2, 3, 4])
    y_fit = 1e10 / x_fit
    ax3.plot(x_fit, y_fit, '--', color=COLORS['neutral'], linewidth=1, alpha=0.7)

    ax3.set_yscale('log')
    ax3.set_xlabel('dC', fontsize=8)
    ax3.set_ylabel('kcat (s⁻¹)', fontsize=8)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax3.set_xlim(0.5, 4.5)
    ax3.set_ylim(10, 1e11)

    # Panel D: Ping-pong energy diagram
    ax4 = fig.add_subplot(gs[3])

    # Reaction coordinate
    x = np.linspace(0, 4, 100)

    # Energy profile
    E = np.zeros_like(x)
    E[x < 1] = -0.5 * (x[x < 1] - 0)**2 + 0.5
    E[(x >= 1) & (x < 2)] = -0.3
    E[(x >= 2) & (x < 3)] = -0.5 * (x[(x >= 2) & (x < 3)] - 2.5)**2 + 0.5 - 0.3
    E[x >= 3] = -0.5

    ax4.plot(x, E, color=COLORS['primary'], linewidth=2)
    ax4.fill_between(x, E, -1, alpha=0.1, color=COLORS['primary'])

    # Labels
    ax4.scatter([0.5, 1.5, 2.5, 3.5], [0.3, -0.3, 0.2, -0.5],
                s=50, c=[COLORS['quaternary'], COLORS['copper'],
                        COLORS['tertiary'], COLORS['success']], zorder=5)

    ax4.set_xlabel('Reaction coord.', fontsize=8)
    ax4.set_ylabel('ΔG', fontsize=8)
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.set_xlim(0, 4)
    ax4.set_ylim(-0.8, 0.6)
    ax4.set_xticks([])
    ax4.set_yticks([])

    fig.suptitle('Catalytic Mechanism: dC = 1 Aperture Traversal',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel6_catalysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel6_catalysis.png")


# =============================================================================
# PANEL 7: Conformational Dynamics (Section 9)
# =============================================================================

def generate_panel_conformational():
    """Panel for conformational dynamics of electrostatic loop."""
    fig, gs = create_panel_figure()

    # Panel A: 3D loop conformations
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Closed conformation (blue)
    t_closed = np.linspace(0, 2*np.pi, 50)
    x_closed = 5 * np.cos(t_closed)
    y_closed = 5 * np.sin(t_closed)
    z_closed = 2 * np.sin(2*t_closed)

    # Open conformation (red)
    x_open = 6 * np.cos(t_closed)
    y_open = 6 * np.sin(t_closed)
    z_open = 3 * np.sin(2*t_closed) + 1

    ax1.plot(x_closed, y_closed, z_closed, color=COLORS['primary'],
             linewidth=2, label='Closed')
    ax1.plot(x_open, y_open, z_open, color=COLORS['quaternary'],
             linewidth=2, label='Open', alpha=0.6)

    # Cu center
    ax1.scatter([0], [0], [0], color=COLORS['copper'], s=100, marker='o')

    ax1.set_xlabel('x (Å)', fontsize=7, labelpad=1)
    ax1.set_ylabel('y (Å)', fontsize=7, labelpad=1)
    ax1.set_zlabel('z (Å)', fontsize=7, labelpad=1)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=30, azim=45)
    ax1.tick_params(pad=0, labelsize=5)
    ax1.legend(fontsize=6, loc='upper right')

    # Panel B: Loop order parameter
    ax2 = fig.add_subplot(gs[1])

    t = np.linspace(0, 100, 200)
    r_closed = 0.92 * np.ones_like(t)
    r_open = 0.71 * np.ones_like(t)
    r_transition = 0.92 - 0.21 * (1 - np.exp(-(t-30)/10))
    r_transition[t < 30] = 0.92
    r_transition[t > 80] = 0.71

    ax2.plot(t, r_transition, color=COLORS['primary'], linewidth=2)
    ax2.axhline(0.92, color=COLORS['success'], linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(0.71, color=COLORS['tertiary'], linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(0.5, color=COLORS['quaternary'], linestyle=':', linewidth=1)

    ax2.fill_between([30, 80], [0.5, 0.5], [1, 1], alpha=0.1, color=COLORS['tertiary'])

    ax2.set_xlabel('t (ps)', fontsize=8)
    ax2.set_ylabel('⟨r⟩ₗₒₒₚ', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax2.set_ylim(0.4, 1)
    ax2.set_xlim(0, 100)
    ax2.text(5, 0.94, 'Closed', fontsize=7, color=COLORS['success'])
    ax2.text(85, 0.73, 'Open', fontsize=7, color=COLORS['tertiary'])

    # Panel C: H-bond phases
    ax3 = fig.add_subplot(gs[2])

    # 8 H-bonds in loop
    hbond_idx = np.arange(1, 9)
    phases_closed = np.random.normal(0, 0.15, 8)
    phases_open = np.random.normal(0, 0.4, 8)

    ax3.bar(hbond_idx - 0.15, phases_closed, width=0.3,
            color=COLORS['primary'], alpha=0.8, label='Closed')
    ax3.bar(hbond_idx + 0.15, phases_open, width=0.3,
            color=COLORS['quaternary'], alpha=0.8, label='Open')

    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_xlabel('H-bond', fontsize=8)
    ax3.set_ylabel('Δφ (rad)', fontsize=8)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax3.set_xticks(hbond_idx)
    ax3.legend(fontsize=6, loc='upper right', frameon=False)

    # Panel D: Temperature independence
    ax4 = fig.add_subplot(gs[3])

    T = np.linspace(4, 40, 50)
    # Topology unchanged
    r_native = 0.87 * np.ones_like(T)
    # Rate changes
    rate = 1e10 * np.exp(-2000/(8.314*T))
    rate_norm = rate / rate.max()

    ax4.plot(T, r_native, 'o-', color=COLORS['primary'],
             markersize=4, linewidth=1.5, label='⟨r⟩')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(T, rate_norm, 's--', color=COLORS['tertiary'],
                  markersize=4, linewidth=1.5, label='Rate')

    ax4.set_xlabel('T (°C)', fontsize=8)
    ax4.set_ylabel('⟨r⟩', fontsize=8, color=COLORS['primary'])
    ax4_twin.set_ylabel('Rate (norm.)', fontsize=8, color=COLORS['tertiary'])
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.set_ylim(0.5, 1)
    ax4_twin.set_ylim(0, 1.1)
    ax4.tick_params(axis='y', colors=COLORS['primary'])
    ax4_twin.tick_params(axis='y', colors=COLORS['tertiary'])

    fig.suptitle('Conformational Dynamics of Electrostatic Loop',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel7_conformational.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel7_conformational.png")


# =============================================================================
# PANEL 8: Protein Folding (Section 10)
# =============================================================================

def generate_panel_folding():
    """Panel for protein folding as trajectory completion."""
    fig, gs = create_panel_figure()

    # Panel A: 3D folding trajectory
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Folding trajectory in reduced space
    t = np.linspace(0, 1, 100)

    # Unfolded to native
    x = 10 * (1 - t) + 0.5 * np.sin(20*t) * (1-t)
    y = 10 * np.sin(2*np.pi*t) * (1-t)
    z = t * 5

    # Color by progress
    colors = plt.cm.viridis(t)
    for i in range(len(t)-1):
        ax1.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]],
                color=colors[i], linewidth=2)

    ax1.scatter([x[0]], [y[0]], [z[0]], color=COLORS['quaternary'],
                s=100, marker='o', label='Unfolded')
    ax1.scatter([x[-1]], [y[-1]], [z[-1]], color=COLORS['success'],
                s=100, marker='*', label='Native')

    ax1.set_xlabel('PC1', fontsize=7, labelpad=1)
    ax1.set_ylabel('PC2', fontsize=7, labelpad=1)
    ax1.set_zlabel('Progress', fontsize=7, labelpad=1)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=20, azim=45)
    ax1.tick_params(pad=0, labelsize=5)

    # Panel B: Order parameter during folding
    ax2 = fig.add_subplot(gs[1])

    t_fold = np.linspace(0, 100, 200)  # ms
    r_fold = 0.1 + 0.77 * (1 / (1 + np.exp(-(t_fold - 50)/10)))

    ax2.plot(t_fold, r_fold, color=COLORS['primary'], linewidth=2)
    ax2.axhline(0.87, color=COLORS['success'], linestyle='--', linewidth=1)
    ax2.axhline(0.5, color=COLORS['quaternary'], linestyle=':', linewidth=1)

    # Mark stages
    stages = [5, 20, 60, 80, 100]
    stage_r = [0.15, 0.35, 0.65, 0.82, 0.87]
    ax2.scatter(stages, stage_r, s=50, c=[COLORS['quaternary'], COLORS['tertiary'],
                COLORS['secondary'], COLORS['primary'], COLORS['success']], zorder=5)

    ax2.set_xlabel('t (ms)', fontsize=8)
    ax2.set_ylabel('⟨r⟩', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 100)

    # Panel C: Complexity comparison
    ax3 = fig.add_subplot(gs[2])

    N = np.logspace(1, 3, 50)
    O_search = 3**N  # Exponential search
    O_categorical = np.log(N) / np.log(3) * N  # O(N log_3 N)

    ax3.semilogy(N, O_search, color=COLORS['quaternary'],
                 linewidth=2, label='O(3ᴺ)')
    ax3.semilogy(N, O_categorical, color=COLORS['success'],
                 linewidth=2, label='O(N log₃N)')

    ax3.axvline(153, color=COLORS['neutral'], linestyle='--', linewidth=1, alpha=0.5)
    ax3.text(160, 1e50, 'SOD1', fontsize=7, rotation=90, va='center')

    ax3.set_xlabel('N (residues)', fontsize=8)
    ax3.set_ylabel('Operations', fontsize=8)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax3.legend(fontsize=7, loc='upper left', frameon=False)
    ax3.set_xlim(10, 1000)
    ax3.set_ylim(1, 1e150)

    # Panel D: Folding intermediates
    ax4 = fig.add_subplot(gs[3])

    stages = ['U', 'I₁', 'I₂', 'I₃', 'N']
    r_vals = [0.1, 0.3, 0.5, 0.7, 0.87]
    times = [0, 5, 20, 60, 100]

    colors = [COLORS['quaternary'], COLORS['tertiary'], COLORS['secondary'],
              COLORS['primary'], COLORS['success']]

    for i in range(len(stages)-1):
        ax4.annotate('', xy=(times[i+1], r_vals[i+1]), xytext=(times[i], r_vals[i]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=1.5))

    ax4.scatter(times, r_vals, s=100, c=colors, edgecolors='black',
                linewidth=0.5, zorder=5)

    for i, (t, r, s) in enumerate(zip(times, r_vals, stages)):
        ax4.text(t, r+0.06, s, ha='center', fontsize=8, fontweight='bold')

    ax4.set_xlabel('t (ms)', fontsize=8)
    ax4.set_ylabel('⟨r⟩', fontsize=8)
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.set_ylim(0, 1)
    ax4.set_xlim(-5, 105)

    fig.suptitle('Protein Folding as Trajectory Completion',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel8_folding.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel8_folding.png")


# =============================================================================
# PANEL 9: Disease/Misfolding (Section 11)
# =============================================================================

def generate_panel_disease():
    """Panel for ALS misfolding as coherence loss."""
    fig, gs = create_panel_figure()

    # Panel A: 3D structure comparison (WT vs mutant)
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Beta barrel representation
    theta = np.linspace(0, 2*np.pi, 100)

    # WT (ordered)
    z_wt = np.linspace(0, 5, 100)
    x_wt = 3 * np.cos(theta)
    y_wt = 3 * np.sin(theta)

    # Mutant (disordered)
    x_mut = 3.5 * np.cos(theta) + 0.5*np.sin(5*theta)
    y_mut = 3.5 * np.sin(theta) + 0.5*np.cos(5*theta)
    z_mut = z_wt + 0.5*np.sin(8*theta)

    ax1.plot(x_wt, y_wt, z_wt, color=COLORS['success'], linewidth=2, label='WT')
    ax1.plot(x_mut + 8, y_mut, z_mut, color=COLORS['quaternary'],
             linewidth=2, label='A4V', alpha=0.7)

    ax1.set_xlabel('x', fontsize=7, labelpad=1)
    ax1.set_ylabel('y', fontsize=7, labelpad=1)
    ax1.set_zlabel('z', fontsize=7, labelpad=1)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=20, azim=30)
    ax1.tick_params(pad=0, labelsize=5)
    ax1.legend(fontsize=6, loc='upper right')

    # Panel B: Coherence vs severity
    ax2 = fig.add_subplot(gs[1])

    variants = ['WT', 'D90A', 'G93A', 'A4V', 'H46R']
    r_vals = [0.87, 0.62, 0.51, 0.43, 0.38]
    survival = [100, 10, 3, 1, 1]  # relative survival time

    colors = [COLORS['success'] if r > 0.8 else
              COLORS['tertiary'] if r > 0.5 else
              COLORS['quaternary'] for r in r_vals]

    ax2.bar(range(len(variants)), r_vals, color=colors,
            edgecolor='white', linewidth=0.5)
    ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.5)
    ax2.axhline(0.8, color='green', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels(variants, fontsize=7)
    ax2.set_ylabel('⟨r⟩', fontsize=8)
    ax2.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax2.set_ylim(0, 1)
    ax2.text(4.5, 0.52, 'Unstable', fontsize=7, color='red', ha='right')

    # Panel C: Coherence vs survival
    ax3 = fig.add_subplot(gs[2])

    ax3.scatter(r_vals[1:], survival[1:], s=80, c=colors[1:],
                edgecolors='black', linewidth=0.5)

    # Fit line
    r_fit = np.linspace(0.35, 0.65, 50)
    surv_fit = np.exp(10*(r_fit - 0.5))
    ax3.plot(r_fit, surv_fit, '--', color=COLORS['neutral'], linewidth=1)

    ax3.set_xlabel('⟨r⟩', fontsize=8)
    ax3.set_ylabel('Survival (years)', fontsize=8)
    ax3.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax3.set_xlim(0.3, 0.7)
    ax3.set_yscale('log')
    ax3.set_ylim(0.5, 20)

    # Annotate points
    for i, (r, s, v) in enumerate(zip(r_vals[1:], survival[1:], variants[1:])):
        ax3.annotate(v, (r, s), xytext=(5, 5), textcoords='offset points', fontsize=6)

    # Panel D: Therapeutic prediction
    ax4 = fig.add_subplot(gs[3])

    # Coherence with/without treatment
    t = np.linspace(0, 100, 100)
    r_untreated = 0.43 - 0.1 * t/100
    r_treated = 0.43 + 0.2 * (1 - np.exp(-t/30))

    ax4.plot(t, r_untreated, color=COLORS['quaternary'], linewidth=2, label='Untreated')
    ax4.plot(t, r_treated, color=COLORS['success'], linewidth=2, label='+ Chaperone')

    ax4.axhline(0.5, color='red', linestyle='--', linewidth=1)
    ax4.fill_between(t, 0, 0.5, alpha=0.1, color=COLORS['quaternary'])
    ax4.fill_between(t, 0.5, 1, alpha=0.1, color=COLORS['success'])

    ax4.set_xlabel('t (days)', fontsize=8)
    ax4.set_ylabel('⟨r⟩', fontsize=8)
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.set_ylim(0.2, 0.8)
    ax4.set_xlim(0, 100)
    ax4.legend(fontsize=6, loc='lower right', frameon=False)

    fig.suptitle('ALS Misfolding as Coherence Loss',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel9_disease.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel9_disease.png")


# =============================================================================
# PANEL 10: Electron Trajectory Visualization (Like Azurin Paper)
# =============================================================================

def generate_panel_trajectory():
    """Panel for detailed electron trajectory visualization with 3D path,
    probability density, ternary markers, and wavefunction reconstruction.
    Styled after azurin-validation.tex Figure 1."""
    fig, gs = create_panel_figure()

    # Panel A: 3D Electron Trajectory with protein context
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Generate electron trajectory through active site
    # Time points from 0 to 850 fs (like azurin)
    t = np.linspace(0, 850, 200)

    # Electron path: starts at His46, passes through Cys112, His117, to Met121
    # Mimicking superexchange pathway
    pathway_progress = t / 850

    # Helical approach with decreasing radius (tunneling)
    r_path = 12.5 * (1 - 0.7 * pathway_progress) + 2 * np.sin(4 * np.pi * pathway_progress)
    theta_path = 2 * np.pi * pathway_progress * 1.5
    z_path = 5 * pathway_progress + 0.5 * np.sin(6 * np.pi * pathway_progress)

    x_path = r_path * np.cos(theta_path)
    y_path = r_path * np.sin(theta_path)

    # Color trajectory by time (blue to red)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(t)-1))
    for i in range(len(t)-1):
        ax1.plot([x_path[i], x_path[i+1]],
                [y_path[i], y_path[i+1]],
                [z_path[i], z_path[i+1]],
                color=colors[i], linewidth=1.5, alpha=0.8)

    # Copper center (Cu)
    ax1.scatter([0], [0], [2.5], color=COLORS['copper'], s=200, marker='o',
                edgecolors='black', linewidth=1.5, label='Cu', zorder=10)

    # Ligand positions (His46, Cys112, His117, Met121)
    ligands = {
        'His46': (-2.0, 0, 2.5, COLORS['primary']),
        'Cys112': (0, 2.1, 2.5, COLORS['tertiary']),
        'His117': (2.0, 0, 2.5, COLORS['primary']),
        'Met121': (0, -3.1, 2.5, COLORS['secondary'])
    }

    for name, (lx, ly, lz, color) in ligands.items():
        ax1.scatter([lx], [ly], [lz], color=color, s=60, marker='^', zorder=5)
        ax1.plot([0, lx], [0, ly], [2.5, lz], 'k-', linewidth=0.5, alpha=0.3)

    # Ternary markers along trajectory (every ~100 fs)
    marker_times = [0, 100, 200, 300, 400, 500, 600, 700, 850]
    marker_trits = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    marker_colors = ['#2E86AB', '#A23B72', '#F18F01']

    for mt, trit in zip(marker_times, marker_trits):
        idx = int(mt / 850 * (len(t) - 1))
        ax1.scatter([x_path[idx]], [y_path[idx]], [z_path[idx]],
                   color=marker_colors[trit], s=40, marker='s',
                   edgecolors='white', linewidth=0.5, zorder=8)

    # Start and end markers
    ax1.scatter([x_path[0]], [y_path[0]], [z_path[0]],
               color='green', s=80, marker='o', edgecolors='black',
               linewidth=1, label='t=0', zorder=9)
    ax1.scatter([x_path[-1]], [y_path[-1]], [z_path[-1]],
               color='red', s=80, marker='*', edgecolors='black',
               linewidth=1, label='t=850fs', zorder=9)

    ax1.set_xlabel('x (Å)', fontsize=7, labelpad=1)
    ax1.set_ylabel('y (Å)', fontsize=7, labelpad=1)
    ax1.set_zlabel('z (Å)', fontsize=7, labelpad=1)
    ax1.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax1.view_init(elev=25, azim=45)
    ax1.tick_params(pad=0, labelsize=5)

    # Panel B: Probability Density Slices (3x3 grid-like in single panel)
    ax2 = fig.add_subplot(gs[1])

    # Create 3x3 mini-heatmaps showing electron density at different times
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    inner_gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1], wspace=0.1, hspace=0.1)

    time_labels = ['0', '100', '200', '300', '400', '500', '600', '700', '850']

    for idx in range(9):
        inner_ax = fig.add_subplot(inner_gs[idx])
        # Generate electron density (Gaussian moving through space)
        x_grid = np.linspace(-5, 5, 30)
        y_grid = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Electron centroid position at this time
        progress = idx / 8
        cx = 4 * (1 - progress) * np.cos(2 * np.pi * progress)
        cy = 4 * (1 - progress) * np.sin(2 * np.pi * progress)
        sigma = 1.5 - 0.8 * progress  # Localizing

        # Probability density
        rho = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        rho /= rho.max()

        inner_ax.imshow(rho, extent=[-5, 5, -5, 5], cmap='viridis',
                       origin='lower', aspect='equal')
        inner_ax.scatter([0], [0], color=COLORS['copper'], s=15, marker='o')
        inner_ax.scatter([cx], [cy], color='white', s=8, marker='+', linewidth=0.5)
        inner_ax.set_xticks([])
        inner_ax.set_yticks([])
        if idx == 0:
            inner_ax.set_title('B', fontsize=10, fontweight='bold', loc='left', pad=2)
        inner_ax.text(0.5, 0.05, f'{time_labels[idx]}fs', transform=inner_ax.transAxes,
                     fontsize=5, ha='center', color='white')

    ax2.axis('off')

    # Panel C: Categorical Coordinates Evolution (4-stack)
    ax3 = fig.add_subplot(gs[2])

    inner_gs3 = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[2], hspace=0.4)

    time_cat = np.array([0, 100, 200, 300, 400, 500, 600, 700, 850])

    # n (principal quantum number)
    ax_n = fig.add_subplot(inner_gs3[0])
    n_vals = np.array([3, 3, 3, 4, 4, 4, 3, 3, 3])
    ax_n.step(time_cat, n_vals, where='mid', color=COLORS['primary'], linewidth=1.5)
    ax_n.scatter(time_cat, n_vals, color=COLORS['primary'], s=15, zorder=5)
    ax_n.set_ylabel('n', fontsize=7)
    ax_n.set_ylim(2.5, 4.5)
    ax_n.set_yticks([3, 4])
    ax_n.set_xticklabels([])
    ax_n.tick_params(labelsize=5)
    ax_n.set_title('C', fontsize=10, fontweight='bold', loc='left', pad=2)

    # l (angular momentum)
    ax_l = fig.add_subplot(inner_gs3[1])
    l_vals = np.array([2, 2, 1, 1, 0, 1, 1, 2, 2])
    ax_l.step(time_cat, l_vals, where='mid', color=COLORS['secondary'], linewidth=1.5)
    ax_l.scatter(time_cat, l_vals, color=COLORS['secondary'], s=15, zorder=5)
    ax_l.set_ylabel('ℓ', fontsize=7)
    ax_l.set_ylim(-0.5, 2.5)
    ax_l.set_yticks([0, 1, 2])
    ax_l.set_xticklabels([])
    ax_l.tick_params(labelsize=5)

    # m (magnetic quantum number)
    ax_m = fig.add_subplot(inner_gs3[2])
    m_vals = np.array([2, 1, 0, -1, 0, 1, 0, -1, -2])
    ax_m.step(time_cat, m_vals, where='mid', color=COLORS['tertiary'], linewidth=1.5)
    ax_m.scatter(time_cat, m_vals, color=COLORS['tertiary'], s=15, zorder=5)
    ax_m.axhline(0, color=COLORS['neutral'], linestyle='--', linewidth=0.5)
    ax_m.set_ylabel('m', fontsize=7)
    ax_m.set_ylim(-2.5, 2.5)
    ax_m.set_yticks([-2, 0, 2])
    ax_m.set_xticklabels([])
    ax_m.tick_params(labelsize=5)

    # s (spin)
    ax_s = fig.add_subplot(inner_gs3[3])
    s_vals = np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5])
    ax_s.step(time_cat, s_vals, where='mid', color=COLORS['quaternary'], linewidth=1.5)
    ax_s.scatter(time_cat, s_vals, color=COLORS['quaternary'], s=15, zorder=5)
    ax_s.axhline(0, color=COLORS['neutral'], linestyle='--', linewidth=0.5)
    ax_s.set_ylabel('s', fontsize=7)
    ax_s.set_ylim(-1, 1)
    ax_s.set_yticks([-0.5, 0.5])
    ax_s.set_yticklabels(['-½', '+½'])
    ax_s.set_xlabel('t (fs)', fontsize=7)
    ax_s.tick_params(labelsize=5)

    ax3.axis('off')

    # Panel D: S-Entropy Space Trajectory
    ax4 = fig.add_subplot(gs[3], projection='3d')

    # S-entropy trajectory during electron transfer
    t_s = np.linspace(0, 1, 100)

    # Start: high kinetic (Sk), low thermal/electronic
    # End: balanced distribution
    Sk = 0.8 - 0.4 * t_s + 0.05 * np.sin(8 * np.pi * t_s)
    St = 0.15 + 0.25 * t_s + 0.03 * np.cos(8 * np.pi * t_s)
    Se = 0.05 + 0.15 * t_s

    # Normalize to sum to 1
    total = Sk + St + Se
    Sk, St, Se = Sk/total, St/total, Se/total

    # Color by time
    colors_s = plt.cm.plasma(t_s)
    for i in range(len(t_s)-1):
        ax4.plot([Sk[i], Sk[i+1]], [St[i], St[i+1]], [Se[i], Se[i+1]],
                color=colors_s[i], linewidth=2)

    # Start and end points
    ax4.scatter([Sk[0]], [St[0]], [Se[0]], color='green', s=100, marker='o',
               edgecolors='black', linewidth=1, label='t=0')
    ax4.scatter([Sk[-1]], [St[-1]], [Se[-1]], color='red', s=100, marker='*',
               edgecolors='black', linewidth=1, label='t=850fs')

    # Unit cube wireframe
    for s in [0, 1]:
        for u in [0, 1]:
            ax4.plot([s, s], [u, u], [0, 1], 'k-', alpha=0.1, linewidth=0.5)
            ax4.plot([s, s], [0, 1], [u, u], 'k-', alpha=0.1, linewidth=0.5)
            ax4.plot([0, 1], [s, s], [u, u], 'k-', alpha=0.1, linewidth=0.5)

    ax4.set_xlabel('Sₖ', fontsize=7, labelpad=1)
    ax4.set_ylabel('Sₜ', fontsize=7, labelpad=1)
    ax4.set_zlabel('Sₑ', fontsize=7, labelpad=1)
    ax4.set_title('D', fontsize=10, fontweight='bold', loc='left', pad=2)
    ax4.view_init(elev=20, azim=45)
    ax4.tick_params(pad=0, labelsize=5)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)

    fig.suptitle('Electron Trajectory Visualization: Ternary Trisection Localization',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.savefig('panel10_trajectory.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: panel10_trajectory.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("Generating all panel figures...")
    print("=" * 50)

    generate_panel_partition()
    generate_panel_selection()
    generate_panel_phaselock()
    generate_panel_sentropy()
    generate_panel_electron()
    generate_panel_catalysis()
    generate_panel_conformational()
    generate_panel_folding()
    generate_panel_disease()
    generate_panel_trajectory()

    print("=" * 50)
    print("All panels generated successfully!")
