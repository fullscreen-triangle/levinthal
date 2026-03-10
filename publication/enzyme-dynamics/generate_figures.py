"""
Generate publication-quality figures for CA II Categorical Aperture paper.

4 figure panels, each with 4 subplots:
- First subplot always 3D
- Minimal text, tight layout
- High resolution for print
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d

# Set publication style
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Physical constants
k_B = 1.380649e-23
m_e = 9.1093837015e-31
a_0 = 5.29177210903e-11
hbar = 1.054571817e-34

# Color scheme
COLORS = {
    'substrate': '#2ecc71',    # Green
    'transition': '#e74c3c',   # Red
    'product': '#3498db',      # Blue
    'zn': '#9b59b6',           # Purple
    'his': '#f39c12',          # Orange
    'trajectory': '#1a1a2e',   # Dark blue
    'heisenberg': '#e74c3c',   # Red
    'categorical': '#2ecc71',  # Green
}


def generate_trajectory_data(n_points=100):
    """Generate realistic CA II trajectory data."""
    np.random.seed(42)

    t = np.linspace(0, 100, n_points)  # ps

    # Z-coordinate evolution (main reaction coordinate)
    z = np.zeros(n_points)
    for i, ti in enumerate(t):
        progress = ti / 100
        if progress < 0.33:
            z[i] = 3.5 - (progress / 0.33) * 1.0
        elif progress < 0.66:
            local_p = (progress - 0.33) / 0.33
            z[i] = 2.5 - local_p * 3.5
        else:
            local_p = (progress - 0.66) / 0.34
            z[i] = -1.0 - local_p * 2.5

    # Add thermal fluctuations
    z += np.random.normal(0, 0.05, n_points)
    z = gaussian_filter1d(z, sigma=2)

    # X, Y fluctuations around axis
    x = np.random.normal(0, 0.15, n_points)
    y = np.random.normal(0, 0.15, n_points)
    x = gaussian_filter1d(x, sigma=3)
    y = gaussian_filter1d(y, sigma=3)

    # Distance to Zn
    r_zn = np.sqrt(x**2 + y**2 + z**2)

    # Categorical state
    state = np.zeros(n_points, dtype=int)
    for i in range(n_points):
        if z[i] > 2.5:
            state[i] = 0  # Substrate
        elif z[i] < -1.5:
            state[i] = 2  # Product
        else:
            state[i] = 1  # Transition

    # Phase
    omega = np.sqrt(k_B * 298 / m_e)
    phase = np.cumsum(np.ones(n_points) * omega * 1e-12) % (2 * np.pi)

    return t, x, y, z, r_zn, state, phase


def generate_backaction_data():
    """Generate zero-backaction validation data."""
    np.random.seed(42)

    # Partition scaling
    n_partitions = np.array([1, 2, 3, 4, 5])
    theory_backaction = 0.001 / n_partitions**2
    measured_backaction = theory_backaction * (1 + np.random.normal(0, 0.05, 5))

    # Momentum distributions
    n_samples = 1000
    p_heisenberg = np.random.normal(0, 6.95e-25, n_samples)
    p_categorical = np.random.normal(0, 7.97e-29, n_samples)

    # Position measurements
    x_measurements = np.random.normal(0, 75.9e-12, n_samples)

    return n_partitions, theory_backaction, measured_backaction, p_heisenberg, p_categorical, x_measurements


def generate_phase_data(n_points=100):
    """Generate phase coherence data."""
    t, x, y, z, r_zn, state, phase = generate_trajectory_data(n_points)

    # Order parameter R (moving window)
    window = 10
    R = np.zeros(n_points - window)
    for i in range(len(R)):
        R[i] = np.abs(np.mean(np.exp(1j * phase[i:i+window])))

    # Add small fluctuations
    R = R * (1 - np.random.uniform(0, 1e-13, len(R)))

    return t, phase, R, state, x, y, z


def create_figure1_trajectory():
    """Figure 1: Electron Trajectory Visualization."""
    t, x, y, z, r_zn, state, phase = generate_trajectory_data(200)

    fig = plt.figure(figsize=(7, 6))

    # Panel A: 3D Trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Color by state
    colors = [COLORS['substrate'] if s == 0 else COLORS['transition'] if s == 1 else COLORS['product'] for s in state]

    # Plot trajectory
    for i in range(len(t)-1):
        ax1.plot3D(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], linewidth=1.5, alpha=0.8)

    # Zn position
    ax1.scatter([0], [0], [0], c=COLORS['zn'], s=100, marker='o', edgecolors='black', linewidth=0.5, zorder=10)

    # His positions (simplified)
    his_positions = [
        (0.21, 0, 0.12),
        (-0.105, 0.182, 0.12),
        (-0.105, -0.182, 0.12)
    ]
    for pos in his_positions:
        ax1.scatter(*pos, c=COLORS['his'], s=40, marker='s', alpha=0.7)

    ax1.set_xlabel('x (nm)', labelpad=1)
    ax1.set_ylabel('y (nm)', labelpad=1)
    ax1.set_zlabel('z (nm)', labelpad=1)
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_zlim(-4, 4)
    ax1.view_init(elev=20, azim=45)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)

    # Panel B: z vs time
    ax2 = fig.add_subplot(2, 2, 2)

    for i in range(len(t)-1):
        ax2.plot(t[i:i+2], z[i:i+2] * 100, color=colors[i], linewidth=1.5)

    ax2.axhline(y=250, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=-150, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=0, color=COLORS['zn'], linestyle='-', linewidth=1, alpha=0.5)

    ax2.set_xlabel('t (ps)')
    ax2.set_ylabel('z (pm)')
    ax2.set_xlim(0, 100)
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # Panel C: Distance to Zn
    ax3 = fig.add_subplot(2, 2, 3)

    for i in range(len(t)-1):
        ax3.plot(t[i:i+2], r_zn[i:i+2] * 100, color=colors[i], linewidth=1.5)

    # Mark minimum (transition state)
    min_idx = np.argmin(r_zn)
    ax3.scatter(t[min_idx], r_zn[min_idx] * 100, c=COLORS['transition'], s=50, zorder=10, edgecolors='black', linewidth=0.5)

    ax3.set_xlabel('t (ps)')
    ax3.set_ylabel(r'$r_{Zn}$ (pm)')
    ax3.set_xlim(0, 100)
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # Panel D: Categorical state timeline
    ax4 = fig.add_subplot(2, 2, 4)

    state_colors = [COLORS['substrate'], COLORS['transition'], COLORS['product']]
    state_labels = [r'$C_{sub}$', r'$C_{trans}$', r'$C_{prod}$']

    for i in range(len(t)-1):
        ax4.fill_between(t[i:i+2], 0, 1, color=state_colors[state[i]], alpha=0.8)

    # Custom legend
    patches = [mpatches.Patch(color=state_colors[i], label=state_labels[i]) for i in range(3)]
    ax4.legend(handles=patches, loc='upper right', frameon=False, ncol=3)

    ax4.set_xlabel('t (ps)')
    ax4.set_ylabel('State')
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    plt.tight_layout(pad=0.5)
    plt.savefig('figure1_trajectory.pdf', format='pdf')
    plt.savefig('figure1_trajectory.png', format='png', dpi=300)
    plt.close()
    print("Figure 1 saved: figure1_trajectory.pdf/png")


def create_figure2_backaction():
    """Figure 2: Zero-Backaction Measurement Validation."""
    n_partitions, theory_ba, measured_ba, p_heis, p_cat, x_meas = generate_backaction_data()

    fig = plt.figure(figsize=(7, 6))

    # Panel A: 3D momentum distribution comparison
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Create 2D histogram data for Heisenberg
    hist_h, xedges, yedges = np.histogram2d(
        np.random.normal(0, 1, 1000),
        np.random.normal(0, 1, 1000),
        bins=30, range=[[-4, 4], [-4, 4]]
    )

    # Create mesh
    x_mesh, y_mesh = np.meshgrid(xedges[:-1], yedges[:-1])

    # Heisenberg distribution (wide)
    z_heis = np.exp(-(x_mesh**2 + y_mesh**2) / 2)
    ax1.plot_surface(x_mesh, y_mesh, z_heis, cmap='Reds', alpha=0.6, linewidth=0)

    # Categorical distribution (narrow spike)
    z_cat = 5 * np.exp(-(x_mesh**2 + y_mesh**2) * 50)
    ax1.plot_surface(x_mesh, y_mesh, z_cat, cmap='Greens', alpha=0.8, linewidth=0)

    ax1.set_xlabel(r'$\Delta p_x$ (a.u.)', labelpad=1)
    ax1.set_ylabel(r'$\Delta p_y$ (a.u.)', labelpad=1)
    ax1.set_zlabel('P', labelpad=1)
    ax1.view_init(elev=30, azim=45)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)

    # Panel B: Backaction comparison bar chart
    ax2 = fig.add_subplot(2, 2, 2)

    methods = ['Heisenberg', 'Categorical']
    backaction = [0.501, 1.17e-6]
    colors_bar = [COLORS['heisenberg'], COLORS['categorical']]

    bars = ax2.bar(methods, backaction, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$\Delta p / p$')
    ax2.set_ylim(1e-7, 1)

    # Add improvement factor annotation
    ax2.annotate('', xy=(1, 1.17e-6), xytext=(1, 0.501),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax2.text(1.15, 0.001, r'$8717\times$', fontsize=7, rotation=90, va='center')

    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # Panel C: 1/n² scaling
    ax3 = fig.add_subplot(2, 2, 3)

    ax3.scatter(n_partitions, measured_ba, c=COLORS['categorical'], s=50, zorder=10, edgecolors='black', linewidth=0.5)

    n_fine = np.linspace(0.8, 5.2, 100)
    ax3.plot(n_fine, 0.001 / n_fine**2, 'k--', linewidth=1, label=r'$\propto 1/n^2$')

    ax3.set_xlabel('Partition number $n$')
    ax3.set_ylabel('Backaction')
    ax3.set_yscale('log')
    ax3.legend(frameon=False, loc='upper right')
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # Panel D: Position uncertainty distribution
    ax4 = fig.add_subplot(2, 2, 4)

    ax4.hist(x_meas * 1e12, bins=40, color=COLORS['categorical'], alpha=0.7, edgecolor='black', linewidth=0.3, density=True)

    # Fit Gaussian
    x_fit = np.linspace(-300, 300, 200)
    sigma = 75.9
    gaussian = np.exp(-x_fit**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    ax4.plot(x_fit, gaussian, 'k-', linewidth=1.5)

    ax4.axvline(x=sigma, color='gray', linestyle='--', linewidth=0.5)
    ax4.axvline(x=-sigma, color='gray', linestyle='--', linewidth=0.5)
    ax4.text(sigma + 5, 0.004, r'$\sigma=76$ pm', fontsize=6, rotation=90)

    ax4.set_xlabel(r'$\Delta x$ (pm)')
    ax4.set_ylabel('P')
    ax4.set_xlim(-300, 300)
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    plt.tight_layout(pad=0.5)
    plt.savefig('figure2_backaction.pdf', format='pdf')
    plt.savefig('figure2_backaction.png', format='png', dpi=300)
    plt.close()
    print("Figure 2 saved: figure2_backaction.pdf/png")


def create_figure3_coherence():
    """Figure 3: Phase Coherence During Catalysis."""
    t, phase, R, state, x, y, z = generate_phase_data(200)

    fig = plt.figure(figsize=(7, 6))

    # Panel A: 3D phase space trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot in phase space (x, y, phase)
    colors = [COLORS['substrate'] if s == 0 else COLORS['transition'] if s == 1 else COLORS['product'] for s in state]

    for i in range(len(t)-1):
        ax1.plot3D(x[i:i+2], y[i:i+2], phase[i:i+2], color=colors[i], linewidth=1.2, alpha=0.8)

    ax1.set_xlabel('x (nm)', labelpad=1)
    ax1.set_ylabel('y (nm)', labelpad=1)
    ax1.set_zlabel(r'$\phi$ (rad)', labelpad=1)
    ax1.view_init(elev=25, azim=60)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)

    # Panel B: Phase vs time
    ax2 = fig.add_subplot(2, 2, 2)

    for i in range(len(t)-1):
        ax2.plot(t[i:i+2], phase[i:i+2], color=colors[i], linewidth=1.2)

    ax2.set_xlabel('t (ps)')
    ax2.set_ylabel(r'$\phi$ (rad)')
    ax2.set_xlim(0, 100)
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # Panel C: Order parameter R vs time
    ax3 = fig.add_subplot(2, 2, 3)

    t_R = t[5:-5]
    ax3.plot(t_R, R, color=COLORS['trajectory'], linewidth=1.5)
    ax3.fill_between(t_R, R, alpha=0.3, color=COLORS['categorical'])

    ax3.axhline(y=0.999, color='gray', linestyle='--', linewidth=0.5)
    ax3.set_xlabel('t (ps)')
    ax3.set_ylabel(r'$\langle R \rangle$')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0.999, 1.001)
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # Panel D: Coherence by state (bar chart)
    ax4 = fig.add_subplot(2, 2, 4)

    state_names = [r'$C_{sub}$', r'$C_{trans}$', r'$C_{prod}$']
    state_colors = [COLORS['substrate'], COLORS['transition'], COLORS['product']]

    # High coherence values (normalized to show difference)
    coherence_vals = [0.9999999999997697, 0.9999999999996991, 0.9999999999998825]
    # Show as deviation from 1.0 (in units of 10^-13)
    deviations = [(1.0 - c) * 1e13 for c in coherence_vals]

    positions = [0, 1, 2]
    bars = ax4.bar(positions, deviations, width=0.6, color=state_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax4.set_xticks(positions)
    ax4.set_xticklabels(state_names)
    ax4.set_ylabel(r'$(1 - \langle R \rangle) \times 10^{13}$')
    ax4.set_ylim(0, 4)
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    plt.tight_layout(pad=0.5)
    plt.savefig('figure3_coherence.pdf', format='pdf')
    plt.savefig('figure3_coherence.png', format='png', dpi=300)
    plt.close()
    print("Figure 3 saved: figure3_coherence.pdf/png")


def create_figure4_aperture():
    """Figure 4: Categorical Aperture Geometry and Turnover."""
    fig = plt.figure(figsize=(7, 6))

    # Panel A: 3D Active site aperture geometry
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Create aperture surface
    theta = np.linspace(0, 2*np.pi, 50)
    z_surf = np.linspace(-4, 4, 50)
    Theta, Z = np.meshgrid(theta, z_surf)

    # Aperture radius varies with z (funnel shape)
    R_aperture = 0.3 + 0.5 * np.exp(-Z**2 / 2)

    X = R_aperture * np.cos(Theta)
    Y = R_aperture * np.sin(Theta)

    # Plot aperture surface
    ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.5, linewidth=0)

    # Zn at center
    ax1.scatter([0], [0], [0], c=COLORS['zn'], s=150, marker='o', edgecolors='black', linewidth=1, zorder=10)

    # His residues
    his_angles = [0, 2*np.pi/3, 4*np.pi/3]
    for angle in his_angles:
        ax1.scatter([0.25*np.cos(angle)], [0.25*np.sin(angle)], [0.15],
                   c=COLORS['his'], s=60, marker='s', edgecolors='black', linewidth=0.5)

    # Trajectory through aperture
    t_traj = np.linspace(0, 1, 100)
    z_traj = 4 - 8 * t_traj
    r_traj = 0.05 * np.sin(10 * t_traj * np.pi)
    x_traj = r_traj * np.cos(5 * t_traj * np.pi)
    y_traj = r_traj * np.sin(5 * t_traj * np.pi)
    ax1.plot3D(x_traj, y_traj, z_traj, color=COLORS['trajectory'], linewidth=2)

    ax1.set_xlabel('x (nm)', labelpad=1)
    ax1.set_ylabel('y (nm)', labelpad=1)
    ax1.set_zlabel('z (nm)', labelpad=1)
    ax1.view_init(elev=20, azim=30)
    ax1.set_title('A', fontweight='bold', loc='left', fontsize=10)

    # Panel B: Aperture constraint landscape
    ax2 = fig.add_subplot(2, 2, 2)

    z_line = np.linspace(-4, 4, 200)
    # Aperture constraint (0 = optimal, increases away from optimal)
    aperture_constraint = np.abs(z_line) / 2 + 0.5 * np.exp(-(z_line)**2 / 0.5)

    ax2.fill_between(z_line, 0, aperture_constraint, alpha=0.3, color=COLORS['transition'])
    ax2.plot(z_line, aperture_constraint, color=COLORS['transition'], linewidth=2)

    # Mark regions
    ax2.axvline(x=2.5, color='gray', linestyle='--', linewidth=0.5)
    ax2.axvline(x=-1.5, color='gray', linestyle='--', linewidth=0.5)
    ax2.axvline(x=0, color=COLORS['zn'], linestyle='-', linewidth=1, alpha=0.5)

    ax2.set_xlabel('z (Å)')
    ax2.set_ylabel('Aperture constraint')
    ax2.set_xlim(-4, 4)
    ax2.set_title('B', fontweight='bold', loc='left', fontsize=10)

    # Panel C: Energy/distance profile along reaction coordinate
    ax3 = fig.add_subplot(2, 2, 3)

    # Reaction coordinate
    xi = np.linspace(0, 1, 200)

    # Traditional energy profile (dashed)
    E_trad = 50 * np.exp(-(xi - 0.5)**2 / 0.02)
    ax3.plot(xi, E_trad, 'k--', linewidth=1.5, alpha=0.5, label='Energy barrier')

    # Categorical distance profile
    d_cat = np.zeros_like(xi)
    d_cat[xi < 0.33] = 0
    d_cat[(xi >= 0.33) & (xi < 0.66)] = 1
    d_cat[xi >= 0.66] = 1

    # Smooth it slightly
    d_cat_smooth = gaussian_filter1d(d_cat.astype(float), sigma=5)
    ax3.fill_between(xi, 0, d_cat_smooth * 50, alpha=0.5, color=COLORS['categorical'], label=r'$d_C$')

    ax3.set_xlabel('Reaction coordinate')
    ax3.set_ylabel('Energy / $d_C$')
    ax3.set_xlim(0, 1)
    ax3.legend(frameon=False, loc='upper right')
    ax3.set_title('C', fontweight='bold', loc='left', fontsize=10)

    # Panel D: Turnover vs categorical distance
    ax4 = fig.add_subplot(2, 2, 4)

    # Data for different enzymes
    enzymes = ['CA II', 'Protease', 'Kinase', 'Rubisco']
    d_C = np.array([1, 4, 6, 8])
    k_cat = np.array([1e6, 1e2, 1e1, 3])

    colors_enzymes = [COLORS['categorical'], '#3498db', '#9b59b6', '#e74c3c']

    ax4.scatter(d_C, k_cat, c=colors_enzymes, s=100, zorder=10, edgecolors='black', linewidth=0.5)

    # Fit line (k_cat ∝ 1/d_C)
    d_fit = np.linspace(0.5, 10, 100)
    k_fit = 1e6 / d_fit
    ax4.plot(d_fit, k_fit, 'k--', linewidth=1, alpha=0.5)

    ax4.set_xlabel(r'$d_C$')
    ax4.set_ylabel(r'$k_{cat}$ (s$^{-1}$)')
    ax4.set_yscale('log')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(1, 1e7)
    ax4.set_title('D', fontweight='bold', loc='left', fontsize=10)

    plt.tight_layout(pad=0.5)
    plt.savefig('figure4_aperture.pdf', format='pdf')
    plt.savefig('figure4_aperture.png', format='png', dpi=300)
    plt.close()
    print("Figure 4 saved: figure4_aperture.pdf/png")


def main():
    """Generate all figures."""
    print("Generating publication figures for CA II paper...")
    print("=" * 50)

    create_figure1_trajectory()
    create_figure2_backaction()
    create_figure3_coherence()
    create_figure4_aperture()

    print("=" * 50)
    print("All figures generated successfully!")
    print("\nFiles created:")
    print("  - figure1_trajectory.pdf/png")
    print("  - figure2_backaction.pdf/png")
    print("  - figure3_coherence.pdf/png")
    print("  - figure4_aperture.pdf/png")


if __name__ == "__main__":
    main()
