#!/usr/bin/env python3
"""
Visualization Panels for Azurin Electron Transfer Experiment

Generates 5 multi-chart visualization panels (4 charts each, including 3D):
1. 3D Trajectory Panel: 3D main + XY/XZ/YZ projections
2. Backaction Panel: 3D surface + line + cumulative + histogram
3. Categorical Panel: 3D quantum state space + time series
4. Probability Panel: 3D isosurface + XY/XZ/YZ slices
5. S-Entropy Panel: 3D trajectory + 3 projections

Author: Kundai Farai Sachikonye
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load validation results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def load_wavefunction(wavefunction_path: Path) -> Dict[str, np.ndarray]:
    """Load wavefunction from NPZ file."""
    data = np.load(wavefunction_path)
    return {
        'psi_real': data['psi_real'],
        'psi_imag': data['psi_imag'],
        'grid_min': data['grid_min'],
        'grid_max': data['grid_max'],
        'resolution': data['resolution']
    }


# =============================================================================
# Panel 1: 3D Trajectory Panel (4 charts: 3D + 3 projections)
# =============================================================================

def plot_panel_1_trajectory(results: Dict[str, Any], output_path: Path) -> plt.Figure:
    """
    Panel 1: 3D Electron Trajectory with projections.

    Layout (2x2):
    - Top-left: 3D trajectory (main view)
    - Top-right: XY projection (top view)
    - Bottom-left: XZ projection (front view)
    - Bottom-right: YZ projection (side view)
    """
    fig = plt.figure(figsize=(16, 14))

    trajectory = results['trajectory']
    positions = np.array([step['position'] for step in trajectory])
    times = np.array([step['time'] for step in trajectory])
    trits = [step['trit'] for step in trajectory]

    # Convert to Angstroms
    positions_A = positions * 1e10
    times_fs = times * 1e15

    # Color normalization
    norm = Normalize(vmin=times_fs.min(), vmax=times_fs.max())
    cmap = plt.cm.coolwarm
    colors = [cmap(norm(t)) for t in times_fs]

    # Trit markers
    trit_colors = {0: 'green', 1: 'gold', 2: 'red'}
    trit_labels = {0: 'Radial', 1: 'Angular', 2: 'Null'}

    # Ligand positions
    ligands = {
        'His46': np.array([2.0, 0.0, 0.0]),
        'Cys112': np.array([0.0, 2.1, 0.0]),
        'His117': np.array([-2.0, 0.0, 0.0]),
        'Met121': np.array([0.0, 0.0, 3.1])
    }

    # ===================== Chart 1: 3D Trajectory (Top-Left) =====================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot trajectory segments colored by time
    for i in range(len(positions_A) - 1):
        ax1.plot(positions_A[i:i+2, 0], positions_A[i:i+2, 1], positions_A[i:i+2, 2],
                color=colors[i], linewidth=2, alpha=0.8)

    # Trit markers every 3rd point
    for i in range(0, len(positions_A), 3):
        ax1.scatter(*positions_A[i], s=60, c=trit_colors[trits[i]],
                   marker='o', edgecolors='black', linewidths=0.5, alpha=0.9)

    # Start/end markers
    ax1.scatter(*positions_A[0], s=200, c='blue', marker='*', edgecolors='black',
               linewidths=1.5, label='Start (Cu+)', zorder=10)
    ax1.scatter(*positions_A[-1], s=200, c='red', marker='*', edgecolors='black',
               linewidths=1.5, label='End (Cu2+)', zorder=10)

    # Copper center
    ax1.scatter(0, 0, 0, s=300, c='orange', marker='o', edgecolors='black',
               linewidths=2, label='Cu center')

    # Ligands
    for name, pos in ligands.items():
        ax1.scatter(*pos, s=100, c='purple', marker='s', edgecolors='black',
                   linewidths=0.5, alpha=0.7)
        ax1.text(pos[0], pos[1], pos[2] + 0.4, name, fontsize=8, ha='center')

    ax1.set_xlabel('X (A)')
    ax1.set_ylabel('Y (A)')
    ax1.set_zlabel('Z (A)')
    ax1.set_title('3D Electron Trajectory', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)

    # ===================== Chart 2: XY Projection (Top-Right) =====================
    ax2 = fig.add_subplot(2, 2, 2)

    # Trajectory
    ax2.scatter(positions_A[:, 0], positions_A[:, 1], c=times_fs, cmap='coolwarm',
               s=30, alpha=0.7, edgecolors='black', linewidths=0.3)
    ax2.plot(positions_A[:, 0], positions_A[:, 1], 'k-', linewidth=0.5, alpha=0.3)

    # Copper and ligands
    ax2.scatter(0, 0, s=200, c='orange', marker='o', edgecolors='black',
               linewidths=2, zorder=10)
    for name, pos in ligands.items():
        ax2.scatter(pos[0], pos[1], s=80, c='purple', marker='s',
                   edgecolors='black', alpha=0.7)
        ax2.annotate(name, (pos[0], pos[1]), fontsize=7, ha='center',
                    xytext=(0, 5), textcoords='offset points')

    # Start/end
    ax2.scatter(positions_A[0, 0], positions_A[0, 1], s=150, c='blue', marker='*',
               edgecolors='black', zorder=10)
    ax2.scatter(positions_A[-1, 0], positions_A[-1, 1], s=150, c='red', marker='*',
               edgecolors='black', zorder=10)

    ax2.set_xlabel('X (A)')
    ax2.set_ylabel('Y (A)')
    ax2.set_title('XY Projection (Top View)', fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.8)
    cbar.set_label('Time (fs)')

    # ===================== Chart 3: XZ Projection (Bottom-Left) =====================
    ax3 = fig.add_subplot(2, 2, 3)

    ax3.scatter(positions_A[:, 0], positions_A[:, 2], c=times_fs, cmap='coolwarm',
               s=30, alpha=0.7, edgecolors='black', linewidths=0.3)
    ax3.plot(positions_A[:, 0], positions_A[:, 2], 'k-', linewidth=0.5, alpha=0.3)

    ax3.scatter(0, 0, s=200, c='orange', marker='o', edgecolors='black',
               linewidths=2, zorder=10)
    for name, pos in ligands.items():
        ax3.scatter(pos[0], pos[2], s=80, c='purple', marker='s',
                   edgecolors='black', alpha=0.7)
        ax3.annotate(name, (pos[0], pos[2]), fontsize=7, ha='center',
                    xytext=(0, 5), textcoords='offset points')

    ax3.scatter(positions_A[0, 0], positions_A[0, 2], s=150, c='blue', marker='*',
               edgecolors='black', zorder=10)
    ax3.scatter(positions_A[-1, 0], positions_A[-1, 2], s=150, c='red', marker='*',
               edgecolors='black', zorder=10)

    ax3.set_xlabel('X (A)')
    ax3.set_ylabel('Z (A)')
    ax3.set_title('XZ Projection (Front View)', fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # ===================== Chart 4: YZ Projection (Bottom-Right) =====================
    ax4 = fig.add_subplot(2, 2, 4)

    ax4.scatter(positions_A[:, 1], positions_A[:, 2], c=times_fs, cmap='coolwarm',
               s=30, alpha=0.7, edgecolors='black', linewidths=0.3)
    ax4.plot(positions_A[:, 1], positions_A[:, 2], 'k-', linewidth=0.5, alpha=0.3)

    ax4.scatter(0, 0, s=200, c='orange', marker='o', edgecolors='black',
               linewidths=2, zorder=10)
    for name, pos in ligands.items():
        ax4.scatter(pos[1], pos[2], s=80, c='purple', marker='s',
                   edgecolors='black', alpha=0.7)
        ax4.annotate(name, (pos[1], pos[2]), fontsize=7, ha='center',
                    xytext=(0, 5), textcoords='offset points')

    ax4.scatter(positions_A[0, 1], positions_A[0, 2], s=150, c='blue', marker='*',
               edgecolors='black', zorder=10)
    ax4.scatter(positions_A[-1, 1], positions_A[-1, 2], s=150, c='red', marker='*',
               edgecolors='black', zorder=10)

    ax4.set_xlabel('Y (A)')
    ax4.set_ylabel('Z (A)')
    ax4.set_title('YZ Projection (Side View)', fontweight='bold')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # Main title
    transfer_time = results['experiment']['transfer_time_fs']
    fig.suptitle(f'Panel 1: 3D Electron Trajectory Through Azurin\n'
                 f'Cu(I) -> Cu(II) Transfer (tau = {transfer_time:.0f} fs)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 2: Backaction Verification Panel (4 charts)
# =============================================================================

def plot_panel_2_backaction(results: Dict[str, Any], output_path: Path) -> plt.Figure:
    """
    Panel 2: Backaction Verification.

    Layout (2x2):
    - Top-left: 3D surface (iteration x position-radius x backaction)
    - Top-right: Backaction vs iteration (line plot)
    - Bottom-left: Cumulative backaction
    - Bottom-right: Backaction distribution histogram
    """
    fig = plt.figure(figsize=(16, 14))

    trajectory = results['trajectory']
    threshold = results['validation']['threshold']

    iterations = np.array([step['iteration'] for step in trajectory])
    backaction = np.array([step['backaction'] for step in trajectory])
    positions = np.array([step['position'] for step in trajectory])

    # Compute radial distance from copper
    radii = np.linalg.norm(positions, axis=1) * 1e10  # in Angstroms

    cumulative = np.cumsum(backaction)

    # ===================== Chart 1: 3D Surface (Top-Left) =====================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Create mesh for surface
    n_iter = len(iterations)
    iter_grid, radius_grid = np.meshgrid(
        np.linspace(iterations.min(), iterations.max(), 20),
        np.linspace(radii.min(), radii.max(), 20)
    )

    # Interpolate backaction values on grid
    from scipy.interpolate import griddata
    points = np.column_stack([iterations, radii])
    backaction_grid = griddata(points, backaction, (iter_grid, radius_grid), method='linear')
    backaction_grid = np.nan_to_num(backaction_grid, nan=np.nanmean(backaction))

    # Plot surface
    surf = ax1.plot_surface(iter_grid, radius_grid, backaction_grid,
                           cmap='viridis', alpha=0.7, edgecolor='none')

    # Scatter actual data points
    ax1.scatter(iterations, radii, backaction, c='red', s=30, alpha=0.8,
               edgecolors='black', linewidths=0.3)

    # Threshold plane
    threshold_plane = np.full_like(iter_grid, threshold)
    ax1.plot_surface(iter_grid, radius_grid, threshold_plane,
                    color='green', alpha=0.2)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Radius (A)')
    ax1.set_zlabel('Backaction (Dp/p)')
    ax1.set_title('3D Backaction Surface', fontweight='bold')

    # ===================== Chart 2: Backaction vs Iteration (Top-Right) =====================
    ax2 = fig.add_subplot(2, 2, 2)

    # Error bars (10% uncertainty)
    backaction_err = backaction * 0.1

    ax2.errorbar(iterations, backaction, yerr=backaction_err,
                fmt='o-', color='royalblue', markersize=5, capsize=2,
                linewidth=1.5, label='Per-step backaction', alpha=0.8)

    # Threshold line
    ax2.axhline(y=threshold, color='green', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold:.0e})')

    # Classical limit
    ax2.axhline(y=1, color='orange', linestyle='--', linewidth=2,
               label='Classical limit')

    # Mean line
    mean_ba = results['metrics']['mean_backaction']
    ax2.axhline(y=mean_ba, color='purple', linestyle=':', linewidth=1.5,
               label=f'Mean ({mean_ba:.2e})')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Backaction (Dp/p)')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-6, 2)
    ax2.set_title('Backaction per Iteration', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # ===================== Chart 3: Cumulative Backaction (Bottom-Left) =====================
    ax3 = fig.add_subplot(2, 2, 3)

    ax3.fill_between(iterations, 0, cumulative, alpha=0.3, color='royalblue')
    ax3.plot(iterations, cumulative, 'b-', linewidth=2, label='Cumulative backaction')

    # Threshold line
    ax3.axhline(y=threshold, color='green', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold:.0e})')

    # Mark points above threshold
    above_threshold = cumulative > threshold
    if any(above_threshold):
        first_above = np.argmax(above_threshold)
        ax3.axvline(x=iterations[first_above], color='red', linestyle=':',
                   linewidth=1.5, label=f'Threshold exceeded at iter {iterations[first_above]}')

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cumulative Backaction (Dp/p)')
    ax3.set_title('Cumulative Backaction', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Final value annotation
    ax3.annotate(f'Total: {cumulative[-1]:.2e}',
                xy=(iterations[-1], cumulative[-1]),
                xytext=(iterations[-1]-3, cumulative[-1]*1.5),
                fontsize=10, ha='right',
                arrowprops=dict(arrowstyle='->', color='black'))

    # ===================== Chart 4: Histogram (Bottom-Right) =====================
    ax4 = fig.add_subplot(2, 2, 4)

    # Histogram of backaction values
    n_bins = min(20, len(backaction))
    counts, bins, patches = ax4.hist(backaction, bins=n_bins, color='steelblue',
                                     edgecolor='black', alpha=0.7)

    # Color bars based on threshold
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge > threshold:
            patch.set_facecolor('coral')

    # Threshold line
    ax4.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold:.0e})')

    # Statistics
    ax4.axvline(x=mean_ba, color='purple', linestyle=':', linewidth=2,
               label=f'Mean ({mean_ba:.2e})')
    ax4.axvline(x=np.median(backaction), color='orange', linestyle=':',
               linewidth=2, label=f'Median ({np.median(backaction):.2e})')

    ax4.set_xlabel('Backaction (Dp/p)')
    ax4.set_ylabel('Count')
    ax4.set_title('Backaction Distribution', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Main title
    verified = results['validation']['zero_backaction_verified']
    status = "PASSED" if verified else "EXCEEDED"
    fig.suptitle(f'Panel 2: Zero-Backaction Verification\n'
                 f'Total Backaction: {cumulative[-1]:.2e} | Status: {status}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 3: Categorical Coordinates Panel (4 charts)
# =============================================================================

def plot_panel_3_categorical(results: Dict[str, Any], output_path: Path) -> plt.Figure:
    """
    Panel 3: Categorical Coordinates Evolution.

    Layout (2x2):
    - Top-left: 3D (n, l, m) quantum state trajectory
    - Top-right: n and l vs time
    - Bottom-left: m and s vs time
    - Bottom-right: Phase space (n vs l colored by time)
    """
    fig = plt.figure(figsize=(16, 14))

    cat_trajectory = results['categorical_trajectory']

    if not cat_trajectory:
        print("Warning: No categorical trajectory data")
        return fig

    times = np.array([state['time'] for state in cat_trajectory]) * 1e15
    n_vals = np.array([state['n'] for state in cat_trajectory])
    l_vals = np.array([state['l'] for state in cat_trajectory])
    m_vals = np.array([state['m'] for state in cat_trajectory])
    s_vals = np.array([state['s'] for state in cat_trajectory])

    norm = Normalize(vmin=times.min(), vmax=times.max())
    cmap = plt.cm.viridis

    # ===================== Chart 1: 3D Quantum State Space (Top-Left) =====================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot trajectory in (n, l, m) space
    scatter = ax1.scatter(n_vals, l_vals, m_vals, c=times, cmap='viridis',
                         s=50, alpha=0.8, edgecolors='black', linewidths=0.3)
    ax1.plot(n_vals, l_vals, m_vals, 'k-', linewidth=0.5, alpha=0.4)

    # Start/end markers
    ax1.scatter(n_vals[0], l_vals[0], m_vals[0], s=200, c='blue', marker='*',
               edgecolors='black', linewidths=1.5, label='Start', zorder=10)
    ax1.scatter(n_vals[-1], l_vals[-1], m_vals[-1], s=200, c='red', marker='*',
               edgecolors='black', linewidths=1.5, label='End', zorder=10)

    ax1.set_xlabel('n (Principal)')
    ax1.set_ylabel('l (Angular)')
    ax1.set_zlabel('m (Magnetic)')
    ax1.set_title('3D Quantum State Trajectory', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.7, pad=0.1)
    cbar.set_label('Time (fs)')

    # ===================== Chart 2: n and l vs Time (Top-Right) =====================
    ax2 = fig.add_subplot(2, 2, 2)

    ax2.plot(times, n_vals, 'o-', color='royalblue', markersize=4,
            linewidth=1.5, label='n (Principal)', alpha=0.8)
    ax2.plot(times, l_vals, 's-', color='forestgreen', markersize=4,
            linewidth=1.5, label='l (Angular)', alpha=0.8)

    # Mark transitions
    n_trans = np.where(np.diff(n_vals) != 0)[0]
    l_trans = np.where(np.diff(l_vals) != 0)[0]
    for t_idx in n_trans:
        ax2.axvline(x=times[t_idx], color='royalblue', linestyle=':', alpha=0.4)
    for t_idx in l_trans:
        ax2.axvline(x=times[t_idx], color='forestgreen', linestyle=':', alpha=0.4)

    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Quantum Number')
    ax2.set_title('n and l Evolution', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ===================== Chart 3: m and s vs Time (Bottom-Left) =====================
    ax3 = fig.add_subplot(2, 2, 3)

    ax3.plot(times, m_vals, 'o-', color='darkorange', markersize=4,
            linewidth=1.5, label='m (Magnetic)', alpha=0.8)
    ax3.plot(times, s_vals, 's-', color='purple', markersize=4,
            linewidth=1.5, label='s (Spin)', alpha=0.8)

    # Mark transitions
    m_trans = np.where(np.diff(m_vals) != 0)[0]
    s_trans = np.where(np.diff(s_vals) != 0)[0]
    for t_idx in m_trans:
        ax3.axvline(x=times[t_idx], color='darkorange', linestyle=':', alpha=0.4)
    for t_idx in s_trans:
        ax3.axvline(x=times[t_idx], color='purple', linestyle=':', alpha=0.4)

    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('Quantum Number')
    ax3.set_title('m and s Evolution', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ===================== Chart 4: Phase Space n vs l (Bottom-Right) =====================
    ax4 = fig.add_subplot(2, 2, 4)

    scatter = ax4.scatter(n_vals, l_vals, c=times, cmap='viridis',
                         s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax4.plot(n_vals, l_vals, 'k-', linewidth=0.5, alpha=0.3)

    # Start/end
    ax4.scatter(n_vals[0], l_vals[0], s=150, c='blue', marker='*',
               edgecolors='black', zorder=10, label='Start')
    ax4.scatter(n_vals[-1], l_vals[-1], s=150, c='red', marker='*',
               edgecolors='black', zorder=10, label='End')

    # Selection rule arrows (Dl = +/- 1)
    ax4.annotate('', xy=(n_vals[-1], l_vals[-1]), xytext=(n_vals[0], l_vals[0]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.5))

    ax4.set_xlabel('n (Principal)')
    ax4.set_ylabel('l (Angular)')
    ax4.set_title('Phase Space (n vs l)', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
    cbar.set_label('Time (fs)')

    # Main title
    n_total_trans = len(n_trans) + len(l_trans) + len(m_trans) + len(s_trans)
    fig.suptitle(f'Panel 3: Categorical Coordinates (n, l, m, s)\n'
                 f'Total Transitions: {n_total_trans}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 4: Probability Density Panel (4 charts)
# =============================================================================

def plot_panel_4_probability(results: Dict[str, Any],
                             wavefunction: Dict[str, np.ndarray],
                             output_path: Path) -> plt.Figure:
    """
    Panel 4: Probability Density Visualization.

    Layout (2x2):
    - Top-left: 3D isosurface at mid-time
    - Top-right: XY slice (z=0)
    - Bottom-left: XZ slice (y=0)
    - Bottom-right: YZ slice (x=0)
    """
    fig = plt.figure(figsize=(16, 14))

    psi_real = wavefunction['psi_real']
    psi_imag = wavefunction['psi_imag']
    grid_min = wavefunction['grid_min']
    grid_max = wavefunction['grid_max']

    # Compute probability density
    rho = psi_real**2 + psi_imag**2

    n_times = rho.shape[0]
    t_mid = n_times // 2

    # Grid coordinates in Angstroms
    nx, ny, nz = rho.shape[1:]
    x = np.linspace(grid_min[0] * 1e10, grid_max[0] * 1e10, nx)
    y = np.linspace(grid_min[1] * 1e10, grid_max[1] * 1e10, ny)
    z = np.linspace(grid_min[2] * 1e10, grid_max[2] * 1e10, nz)

    X_xy, Y_xy = np.meshgrid(x, y, indexing='ij')
    X_xz, Z_xz = np.meshgrid(x, z, indexing='ij')
    Y_yz, Z_yz = np.meshgrid(y, z, indexing='ij')

    transfer_time = results['experiment']['transfer_time_fs']
    t_fs_mid = t_mid / n_times * transfer_time

    # ===================== Chart 1: 3D Isosurface (Top-Left) =====================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    rho_mid = rho[t_mid]
    rho_max = rho_mid.max()

    # Create isosurface using contour at multiple levels
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Plot slices through the density
    levels = [0.3, 0.5, 0.7]
    colors_iso = ['blue', 'green', 'red']
    alphas = [0.2, 0.4, 0.6]

    X_3d, Y_3d, Z_3d = np.meshgrid(x, y, z, indexing='ij')

    for level, color, alpha in zip(levels, colors_iso, alphas):
        threshold = level * rho_max
        mask = rho_mid > threshold
        ax1.scatter(X_3d[mask], Y_3d[mask], Z_3d[mask],
                   c=color, alpha=alpha, s=1, label=f'{level*100:.0f}% max')

    # Copper center
    ax1.scatter(0, 0, 0, s=200, c='orange', marker='o', edgecolors='black',
               linewidths=2, label='Cu center', zorder=10)

    ax1.set_xlabel('X (A)')
    ax1.set_ylabel('Y (A)')
    ax1.set_zlabel('Z (A)')
    ax1.set_title(f'3D Probability Density (t = {t_fs_mid:.0f} fs)', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)

    # ===================== Chart 2: XY Slice (Top-Right) =====================
    ax2 = fig.add_subplot(2, 2, 2)

    z_idx = nz // 2
    rho_xy = rho_mid[:, :, z_idx]

    im2 = ax2.imshow(rho_xy.T, origin='lower', cmap='hot',
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    aspect='equal', interpolation='bilinear')
    ax2.contour(X_xy, Y_xy, rho_xy, levels=5, colors='white', linewidths=0.5, alpha=0.5)

    # Copper marker
    ax2.plot(0, 0, 'co', markersize=12, markeredgecolor='cyan', markeredgewidth=2)

    ax2.set_xlabel('X (A)')
    ax2.set_ylabel('Y (A)')
    ax2.set_title('XY Slice (z = 0)', fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='rho(r)')

    # ===================== Chart 3: XZ Slice (Bottom-Left) =====================
    ax3 = fig.add_subplot(2, 2, 3)

    y_idx = ny // 2
    rho_xz = rho_mid[:, y_idx, :]

    im3 = ax3.imshow(rho_xz.T, origin='lower', cmap='hot',
                    extent=[x.min(), x.max(), z.min(), z.max()],
                    aspect='equal', interpolation='bilinear')
    ax3.contour(X_xz, Z_xz, rho_xz, levels=5, colors='white', linewidths=0.5, alpha=0.5)

    ax3.plot(0, 0, 'co', markersize=12, markeredgecolor='cyan', markeredgewidth=2)

    ax3.set_xlabel('X (A)')
    ax3.set_ylabel('Z (A)')
    ax3.set_title('XZ Slice (y = 0)', fontweight='bold')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='rho(r)')

    # ===================== Chart 4: YZ Slice (Bottom-Right) =====================
    ax4 = fig.add_subplot(2, 2, 4)

    x_idx = nx // 2
    rho_yz = rho_mid[x_idx, :, :]

    im4 = ax4.imshow(rho_yz.T, origin='lower', cmap='hot',
                    extent=[y.min(), y.max(), z.min(), z.max()],
                    aspect='equal', interpolation='bilinear')
    ax4.contour(Y_yz, Z_yz, rho_yz, levels=5, colors='white', linewidths=0.5, alpha=0.5)

    ax4.plot(0, 0, 'co', markersize=12, markeredgecolor='cyan', markeredgewidth=2)

    ax4.set_xlabel('Y (A)')
    ax4.set_ylabel('Z (A)')
    ax4.set_title('YZ Slice (x = 0)', fontweight='bold')
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='rho(r)')

    # Main title
    fig.suptitle(f'Panel 4: Electron Probability Density\n'
                 f'Snapshot at t = {t_fs_mid:.0f} fs (mid-transfer)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 5: S-Entropy Space Panel (4 charts)
# =============================================================================

def plot_panel_5_sentropy(results: Dict[str, Any], output_path: Path) -> plt.Figure:
    """
    Panel 5: S-Entropy Space Trajectory.

    Layout (2x2):
    - Top-left: 3D trajectory in (S_k, S_t, S_e) space
    - Top-right: S_k vs S_t projection
    - Bottom-left: S_k vs S_e projection
    - Bottom-right: S_t vs S_e projection
    """
    fig = plt.figure(figsize=(16, 14))

    cat_trajectory = results['categorical_trajectory']

    if not cat_trajectory:
        print("Warning: No categorical trajectory data")
        return fig

    s_k = np.array([state['S_k'] for state in cat_trajectory])
    s_t = np.array([state['S_t'] for state in cat_trajectory])
    s_e = np.array([state['S_e'] for state in cat_trajectory])
    times = np.array([state['time'] for state in cat_trajectory]) * 1e15

    norm = Normalize(vmin=times.min(), vmax=times.max())
    cmap = plt.cm.coolwarm

    # ===================== Chart 1: 3D S-Entropy Trajectory (Top-Left) =====================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    scatter = ax1.scatter(s_k, s_t, s_e, c=times, cmap='coolwarm',
                         s=50, alpha=0.8, edgecolors='black', linewidths=0.3)
    ax1.plot(s_k, s_t, s_e, 'k-', linewidth=0.5, alpha=0.4)

    # Start/end
    ax1.scatter(s_k[0], s_t[0], s_e[0], s=200, c='blue', marker='*',
               edgecolors='black', linewidths=1.5, label='Start', zorder=10)
    ax1.scatter(s_k[-1], s_t[-1], s_e[-1], s=200, c='red', marker='*',
               edgecolors='black', linewidths=1.5, label='End', zorder=10)

    # Unit cube wireframe
    for i in [0, 1]:
        for j in [0, 1]:
            ax1.plot([0, 1], [i, i], [j, j], 'gray', linewidth=0.5, alpha=0.3)
            ax1.plot([i, i], [0, 1], [j, j], 'gray', linewidth=0.5, alpha=0.3)
            ax1.plot([i, i], [j, j], [0, 1], 'gray', linewidth=0.5, alpha=0.3)

    ax1.set_xlabel('S_k (Knowledge)')
    ax1.set_ylabel('S_t (Temporal)')
    ax1.set_zlabel('S_e (Evolution)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_title('3D S-Entropy Trajectory', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)

    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.7, pad=0.1)
    cbar.set_label('Time (fs)')

    # ===================== Chart 2: S_k vs S_t (Top-Right) =====================
    ax2 = fig.add_subplot(2, 2, 2)

    scatter2 = ax2.scatter(s_k, s_t, c=times, cmap='coolwarm',
                          s=50, alpha=0.8, edgecolors='black', linewidths=0.3)
    ax2.plot(s_k, s_t, 'k-', linewidth=0.5, alpha=0.3)

    ax2.scatter(s_k[0], s_t[0], s=150, c='blue', marker='*',
               edgecolors='black', zorder=10, label='Start')
    ax2.scatter(s_k[-1], s_t[-1], s=150, c='red', marker='*',
               edgecolors='black', zorder=10, label='End')

    ax2.set_xlabel('S_k (Knowledge Entropy)')
    ax2.set_ylabel('S_t (Temporal Entropy)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('S_k vs S_t Projection', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.colorbar(scatter2, ax=ax2, shrink=0.8, label='Time (fs)')

    # ===================== Chart 3: S_k vs S_e (Bottom-Left) =====================
    ax3 = fig.add_subplot(2, 2, 3)

    scatter3 = ax3.scatter(s_k, s_e, c=times, cmap='coolwarm',
                          s=50, alpha=0.8, edgecolors='black', linewidths=0.3)
    ax3.plot(s_k, s_e, 'k-', linewidth=0.5, alpha=0.3)

    ax3.scatter(s_k[0], s_e[0], s=150, c='blue', marker='*',
               edgecolors='black', zorder=10, label='Start')
    ax3.scatter(s_k[-1], s_e[-1], s=150, c='red', marker='*',
               edgecolors='black', zorder=10, label='End')

    ax3.set_xlabel('S_k (Knowledge Entropy)')
    ax3.set_ylabel('S_e (Evolution Entropy)')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('S_k vs S_e Projection', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    plt.colorbar(scatter3, ax=ax3, shrink=0.8, label='Time (fs)')

    # ===================== Chart 4: S_t vs S_e (Bottom-Right) =====================
    ax4 = fig.add_subplot(2, 2, 4)

    scatter4 = ax4.scatter(s_t, s_e, c=times, cmap='coolwarm',
                          s=50, alpha=0.8, edgecolors='black', linewidths=0.3)
    ax4.plot(s_t, s_e, 'k-', linewidth=0.5, alpha=0.3)

    ax4.scatter(s_t[0], s_e[0], s=150, c='blue', marker='*',
               edgecolors='black', zorder=10, label='Start')
    ax4.scatter(s_t[-1], s_e[-1], s=150, c='red', marker='*',
               edgecolors='black', zorder=10, label='End')

    ax4.set_xlabel('S_t (Temporal Entropy)')
    ax4.set_ylabel('S_e (Evolution Entropy)')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('S_t vs S_e Projection', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    plt.colorbar(scatter4, ax=ax4, shrink=0.8, label='Time (fs)')

    # Path length
    path_length = sum(np.sqrt((s_k[i+1] - s_k[i])**2 +
                              (s_t[i+1] - s_t[i])**2 +
                              (s_e[i+1] - s_e[i])**2)
                      for i in range(len(s_k) - 1))

    # Main title
    fig.suptitle(f'Panel 5: S-Entropy Space Trajectory\n'
                 f'Path Length: {path_length:.3f} | States: {len(cat_trajectory)}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 6: Protein Structure Panel (4 3D charts)
# =============================================================================

def plot_panel_6_protein_structure(results: Dict[str, Any], output_path: Path) -> plt.Figure:
    """
    Panel 6: Protein Structure with Electron Trajectory.

    Layout (2x2) - ALL 3D CHARTS:
    - Top-left: Full protein structure with trajectory (overview)
    - Top-right: Copper active site zoom with ligands
    - Bottom-left: Protein backbone/ribbon representation
    - Bottom-right: Multi-angle view with trajectory time evolution
    """
    fig = plt.figure(figsize=(18, 16))

    trajectory = results['trajectory']
    positions = np.array([step['position'] for step in trajectory])
    times = np.array([step['time'] for step in trajectory])
    trits = [step['trit'] for step in trajectory]

    # Convert to Angstroms
    positions_A = positions * 1e10
    times_fs = times * 1e15

    # Color by time
    norm = Normalize(vmin=times_fs.min(), vmax=times_fs.max())
    cmap = plt.cm.plasma

    # Azurin protein structure (simplified representation)
    # Approximate backbone coordinates for azurin (128 residues)
    np.random.seed(42)  # Reproducible
    n_residues = 128

    # Create helical backbone structure typical of blue copper proteins
    theta = np.linspace(0, 8 * np.pi, n_residues)
    r_backbone = 8 + 3 * np.sin(theta * 0.5)  # Varying radius
    z_backbone = np.linspace(-15, 15, n_residues) + 2 * np.sin(theta)

    backbone_x = r_backbone * np.cos(theta)
    backbone_y = r_backbone * np.sin(theta)
    backbone_z = z_backbone

    # Beta sheet regions (characteristic of azurin)
    beta_sheet_1 = slice(20, 35)
    beta_sheet_2 = slice(45, 60)
    beta_sheet_3 = slice(80, 95)
    beta_sheet_4 = slice(100, 115)

    # Ligand positions (copper coordination)
    ligands = {
        'His46': np.array([2.0, 0.0, 0.0]),
        'Cys112': np.array([0.0, 2.1, 0.0]),
        'His117': np.array([-2.0, 0.0, 0.0]),
        'Met121': np.array([0.0, 0.0, 3.1])
    }

    # ===================== Chart 1: Full Protein Overview (Top-Left) =====================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Draw backbone as tubes
    ax1.plot(backbone_x, backbone_y, backbone_z, 'gray', linewidth=2, alpha=0.6,
            label='Backbone')

    # Highlight beta sheets
    for sheet, color in [(beta_sheet_1, 'royalblue'), (beta_sheet_2, 'forestgreen'),
                         (beta_sheet_3, 'darkorange'), (beta_sheet_4, 'purple')]:
        ax1.plot(backbone_x[sheet], backbone_y[sheet], backbone_z[sheet],
                color=color, linewidth=4, alpha=0.8)

    # Draw side chains as small spheres (sampled)
    for i in range(0, n_residues, 5):
        # Random side chain direction
        sc_dir = np.random.randn(3)
        sc_dir = sc_dir / np.linalg.norm(sc_dir) * 2
        sc_pos = np.array([backbone_x[i], backbone_y[i], backbone_z[i]]) + sc_dir
        ax1.scatter(*sc_pos, s=20, c='lightgray', alpha=0.4)

    # Copper center (at origin)
    ax1.scatter(0, 0, 0, s=400, c='orange', marker='o', edgecolors='black',
               linewidths=2, label='Cu center', zorder=10)

    # Ligands
    for name, pos in ligands.items():
        ax1.scatter(*pos, s=150, c='purple', marker='s', edgecolors='black',
                   linewidths=1, alpha=0.9)
        ax1.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 'purple', linewidth=1.5, alpha=0.7)

    # Electron trajectory
    for i in range(len(positions_A) - 1):
        color = cmap(norm(times_fs[i]))
        ax1.plot(positions_A[i:i+2, 0], positions_A[i:i+2, 1], positions_A[i:i+2, 2],
                color=color, linewidth=3, alpha=0.9)

    # Start/end markers
    ax1.scatter(*positions_A[0], s=250, c='cyan', marker='*', edgecolors='black',
               linewidths=1.5, label='e- Start', zorder=15)
    ax1.scatter(*positions_A[-1], s=250, c='red', marker='*', edgecolors='black',
               linewidths=1.5, label='e- End', zorder=15)

    ax1.set_xlabel('X (A)')
    ax1.set_ylabel('Y (A)')
    ax1.set_zlabel('Z (A)')
    ax1.set_title('Full Protein Structure with Trajectory', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=8)

    # ===================== Chart 2: Active Site Zoom (Top-Right) =====================
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')

    # Copper center
    ax2.scatter(0, 0, 0, s=600, c='orange', marker='o', edgecolors='black',
               linewidths=3, label='Cu(I/II)', zorder=10)

    # Ligand atoms with labels and bonds
    ligand_colors = {'His46': 'blue', 'Cys112': 'yellow', 'His117': 'blue', 'Met121': 'green'}

    for name, pos in ligands.items():
        color = ligand_colors[name]
        # Main coordinating atom
        ax2.scatter(*pos, s=200, c=color, marker='o', edgecolors='black',
                   linewidths=1.5, alpha=0.9)
        # Bond to copper
        ax2.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 'k-', linewidth=2, alpha=0.8)
        # Label
        ax2.text(pos[0]*1.3, pos[1]*1.3, pos[2]*1.3, name, fontsize=10,
                fontweight='bold', ha='center')

        # Draw residue ring/structure
        if 'His' in name:
            # Imidazole ring
            ring_angles = np.linspace(0, 2*np.pi, 6)
            ring_r = 0.8
            ring_center = pos * 1.5
            ring_x = ring_center[0] + ring_r * np.cos(ring_angles)
            ring_y = ring_center[1] + ring_r * np.sin(ring_angles)
            ring_z = np.full_like(ring_angles, ring_center[2])
            ax2.plot(ring_x, ring_y, ring_z, color, linewidth=2, alpha=0.7)
        elif 'Cys' in name:
            # Thiolate
            ax2.scatter(pos[0]*1.3, pos[1]*1.3, pos[2]*1.3, s=80, c='yellow',
                       marker='o', alpha=0.7)

    # Electron trajectory in active site
    for i in range(len(positions_A) - 1):
        color = cmap(norm(times_fs[i]))
        ax2.plot(positions_A[i:i+2, 0], positions_A[i:i+2, 1], positions_A[i:i+2, 2],
                color=color, linewidth=4, alpha=0.9)

    # Trajectory points
    ax2.scatter(positions_A[:, 0], positions_A[:, 1], positions_A[:, 2],
               c=times_fs, cmap='plasma', s=40, alpha=0.7, edgecolors='black', linewidths=0.3)

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_zlim(-5, 5)
    ax2.set_xlabel('X (A)')
    ax2.set_ylabel('Y (A)')
    ax2.set_zlabel('Z (A)')
    ax2.set_title('Copper Active Site (Zoomed)', fontweight='bold', fontsize=12)

    # ===================== Chart 3: Backbone Ribbon (Bottom-Left) =====================
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')

    # Draw ribbon representation
    # Create ribbon width perpendicular to backbone
    ribbon_width = 1.5

    for i in range(n_residues - 1):
        # Direction along backbone
        tangent = np.array([backbone_x[i+1] - backbone_x[i],
                           backbone_y[i+1] - backbone_y[i],
                           backbone_z[i+1] - backbone_z[i]])
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)

        # Perpendicular direction (for ribbon width)
        up = np.array([0, 0, 1])
        perp = np.cross(tangent, up)
        perp = perp / (np.linalg.norm(perp) + 1e-10) * ribbon_width

        # Color by secondary structure
        if i in range(20, 35) or i in range(45, 60) or i in range(80, 95) or i in range(100, 115):
            color = 'royalblue'  # Beta sheet
            width = 2.5
        else:
            color = 'lightgray'  # Coil
            width = 1.0

        ax3.plot([backbone_x[i], backbone_x[i+1]],
                [backbone_y[i], backbone_y[i+1]],
                [backbone_z[i], backbone_z[i+1]],
                color=color, linewidth=width, alpha=0.8)

    # Mark N and C termini
    ax3.scatter(backbone_x[0], backbone_y[0], backbone_z[0], s=200, c='green',
               marker='^', edgecolors='black', linewidths=1.5, label='N-terminus')
    ax3.scatter(backbone_x[-1], backbone_y[-1], backbone_z[-1], s=200, c='red',
               marker='v', edgecolors='black', linewidths=1.5, label='C-terminus')

    # Copper center
    ax3.scatter(0, 0, 0, s=400, c='orange', marker='o', edgecolors='black',
               linewidths=2, label='Cu center')

    # Electron trajectory
    for i in range(len(positions_A) - 1):
        color = cmap(norm(times_fs[i]))
        ax3.plot(positions_A[i:i+2, 0], positions_A[i:i+2, 1], positions_A[i:i+2, 2],
                color=color, linewidth=3, alpha=0.9)

    ax3.set_xlabel('X (A)')
    ax3.set_ylabel('Y (A)')
    ax3.set_zlabel('Z (A)')
    ax3.set_title('Protein Backbone (Ribbon)', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper left', fontsize=8)

    # ===================== Chart 4: Time Evolution Multi-View (Bottom-Right) =====================
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    # Show trajectory with time-coded markers and velocity vectors
    # Copper center
    ax4.scatter(0, 0, 0, s=400, c='orange', marker='o', edgecolors='black',
               linewidths=2, label='Cu center', zorder=5)

    # Ligands (semi-transparent)
    for name, pos in ligands.items():
        ax4.scatter(*pos, s=100, c='purple', marker='s', edgecolors='black',
                   linewidths=0.5, alpha=0.5)
        ax4.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 'purple', linewidth=1, alpha=0.3)

    # Trajectory with velocity arrows
    arrow_interval = max(1, len(positions_A) // 8)  # Show ~8 arrows

    for i in range(len(positions_A)):
        # Point colored by time
        color = cmap(norm(times_fs[i]))
        size = 60 + 40 * (i / len(positions_A))  # Increasing size
        ax4.scatter(positions_A[i, 0], positions_A[i, 1], positions_A[i, 2],
                   c=[color], s=size, alpha=0.8, edgecolors='black', linewidths=0.3)

        # Velocity arrows at intervals
        if i % arrow_interval == 0 and i < len(positions_A) - 1:
            vel = positions_A[i+1] - positions_A[i]
            vel_norm = vel / (np.linalg.norm(vel) + 1e-10) * 0.5  # Normalize arrow length
            ax4.quiver(positions_A[i, 0], positions_A[i, 1], positions_A[i, 2],
                      vel_norm[0], vel_norm[1], vel_norm[2],
                      color=color, arrow_length_ratio=0.3, linewidth=2, alpha=0.8)

    # Connect trajectory
    ax4.plot(positions_A[:, 0], positions_A[:, 1], positions_A[:, 2],
            'k-', linewidth=1, alpha=0.3)

    # Time labels at key points
    time_labels = [0, len(positions_A)//4, len(positions_A)//2, 3*len(positions_A)//4, len(positions_A)-1]
    for idx in time_labels:
        if idx < len(positions_A):
            ax4.text(positions_A[idx, 0] + 0.3, positions_A[idx, 1] + 0.3,
                    positions_A[idx, 2] + 0.3, f'{times_fs[idx]:.0f}fs',
                    fontsize=8, color='black')

    # Start/end emphasis
    ax4.scatter(*positions_A[0], s=300, c='cyan', marker='*', edgecolors='black',
               linewidths=2, label='Start (0 fs)', zorder=10)
    ax4.scatter(*positions_A[-1], s=300, c='red', marker='*', edgecolors='black',
               linewidths=2, label=f'End ({times_fs[-1]:.0f} fs)', zorder=10)

    ax4.set_xlabel('X (A)')
    ax4.set_ylabel('Y (A)')
    ax4.set_zlabel('Z (A)')
    ax4.set_title('Time Evolution with Velocity', fontweight='bold', fontsize=12)
    ax4.legend(loc='upper left', fontsize=8)

    # Add colorbar for time
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Time (fs)', fontsize=11)

    # Main title
    transfer_time = results['experiment']['transfer_time_fs']
    fig.suptitle(f'Panel 6: Azurin Protein Structure with Electron Trajectory\n'
                 f'PDB: 4AZU | Cu(I) -> Cu(II) Transfer | tau = {transfer_time:.0f} fs',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 7: Wavefunction Analysis Panel (4 charts)
# =============================================================================

def plot_panel_7_wavefunction(wavefunction: Dict[str, np.ndarray], results: Dict[str, Any],
                               output_path: Path) -> plt.Figure:
    """
    Panel 7: Wavefunction Analysis and Evolution.

    Layout (2x2):
    - Top-left: 3D probability density isosurface at peak time
    - Top-right: Time evolution of probability density (waterfall plot)
    - Bottom-left: Phase coherence map (real vs imaginary)
    - Bottom-right: Spatial localization metrics over time
    """
    fig = plt.figure(figsize=(18, 16))

    psi_real = wavefunction['psi_real']
    psi_imag = wavefunction['psi_imag']
    grid_min = wavefunction['grid_min']
    grid_max = wavefunction['grid_max']
    resolution = float(wavefunction['resolution'])

    n_times, nx, ny, nz = psi_real.shape

    # Compute probability density |ψ|²
    prob_density = psi_real**2 + psi_imag**2

    # Compute phase angle
    phase = np.arctan2(psi_imag, psi_real)

    # Create coordinate arrays (in Angstroms)
    x = np.linspace(grid_min[0], grid_max[0], nx) * 1e10
    y = np.linspace(grid_min[1], grid_max[1], ny) * 1e10
    z = np.linspace(grid_min[2], grid_max[2], nz) * 1e10

    # Get time array
    trajectory = results.get('trajectory', [])
    if trajectory:
        times_fs = np.array([step['time'] for step in trajectory[:n_times]]) * 1e15
    else:
        times_fs = np.arange(n_times) * 10.0  # Default 10 fs spacing

    # ===================== Chart 1: 3D Probability Isosurface (Top-Left) =====================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Find timestep with maximum integrated probability (peak localization)
    total_prob = np.sum(prob_density, axis=(1, 2, 3))
    peak_time_idx = np.argmax(total_prob)

    # Get probability at peak time
    prob_at_peak = prob_density[peak_time_idx]

    # Find isosurface threshold (50% of max)
    threshold = 0.3 * np.max(prob_at_peak)

    # Extract isosurface points (simplified: show high-probability voxels)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Sample high-probability points for visualization
    high_prob_mask = prob_at_peak > threshold
    if np.any(high_prob_mask):
        # Downsample for performance
        indices = np.where(high_prob_mask)
        n_points = min(2000, len(indices[0]))
        sample_idx = np.random.choice(len(indices[0]), n_points, replace=False)

        scatter_x = X[high_prob_mask][sample_idx]
        scatter_y = Y[high_prob_mask][sample_idx]
        scatter_z = Z[high_prob_mask][sample_idx]
        scatter_prob = prob_at_peak[high_prob_mask][sample_idx]

        # Normalize for coloring
        scatter_prob_norm = scatter_prob / scatter_prob.max()

        sc = ax1.scatter(scatter_x, scatter_y, scatter_z,
                        c=scatter_prob_norm, cmap='hot', s=20,
                        alpha=0.6, edgecolors='none')
        plt.colorbar(sc, ax=ax1, label='|ψ|² (normalized)', shrink=0.6)
    else:
        # Fallback: show center slice
        mid_z = nz // 2
        ax1.text(0, 0, 0, 'Low probability\n(see slices)', ha='center', fontsize=10)

    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title(f'Probability Density Isosurface\nt = {times_fs[peak_time_idx]:.1f} fs (peak)',
                  fontweight='bold', fontsize=12)

    # ===================== Chart 2: Time Evolution Waterfall (Top-Right) =====================
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')

    # Compute integrated probability along z for each time (2D slices)
    # This gives a "movie" of probability evolution
    prob_xy = np.sum(prob_density, axis=3)  # Integrate over z

    # Take central slice in y for waterfall plot
    mid_y = ny // 2
    prob_xt = prob_xy[:, :, mid_y]  # Shape: (n_times, nx)

    # Create waterfall plot
    X_wf, T_wf = np.meshgrid(x, times_fs[:n_times])

    # Normalize each time slice for visibility
    prob_xt_norm = prob_xt / (prob_xt.max(axis=1, keepdims=True) + 1e-10)

    # Plot as surface
    surf = ax2.plot_surface(X_wf, T_wf, prob_xt_norm,
                            cmap='viridis', alpha=0.8,
                            linewidth=0, antialiased=True)

    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Time (fs)')
    ax2.set_zlabel('|ψ|² (normalized)')
    ax2.set_title('Probability Evolution (X-T slice)\nIntegrated over Y,Z',
                  fontweight='bold', fontsize=12)
    ax2.view_init(elev=25, azim=-60)

    # ===================== Chart 3: Phase Coherence Map (Bottom-Left) =====================
    ax3 = fig.add_subplot(2, 2, 3)

    # Compute phase coherence: correlation between real and imaginary parts
    # High coherence = well-defined phase, Low coherence = decoherence

    # Take central XY slice at each time
    mid_z = nz // 2

    # Create phase coherence metric: |<exp(i*phase)>| averaged over space
    coherence = np.zeros(n_times)
    mean_phase = np.zeros(n_times)
    phase_std = np.zeros(n_times)

    for t in range(n_times):
        phase_slice = phase[t, :, :, mid_z]
        prob_slice = prob_density[t, :, :, mid_z]

        # Weighted average phase (by probability)
        if np.sum(prob_slice) > 0:
            weights = prob_slice / np.sum(prob_slice)
            exp_phase = np.exp(1j * phase_slice)
            coherence[t] = np.abs(np.sum(weights * exp_phase))
            mean_phase[t] = np.angle(np.sum(weights * exp_phase))
            phase_std[t] = np.sqrt(np.sum(weights * (phase_slice - mean_phase[t])**2))
        else:
            coherence[t] = 0
            mean_phase[t] = 0
            phase_std[t] = np.pi

    # Plot coherence and phase
    ax3_twin = ax3.twinx()

    line1, = ax3.plot(times_fs[:n_times], coherence, 'b-', linewidth=2.5,
                      marker='o', markersize=6, label='Coherence |⟨e^{iφ}⟩|')
    ax3.fill_between(times_fs[:n_times], 0, coherence, alpha=0.3, color='blue')
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Coherence threshold')

    line2, = ax3_twin.plot(times_fs[:n_times], mean_phase, 'r-', linewidth=2,
                           marker='s', markersize=5, label='Mean phase ⟨φ⟩')
    ax3_twin.fill_between(times_fs[:n_times],
                          mean_phase - phase_std, mean_phase + phase_std,
                          alpha=0.2, color='red')

    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('Phase Coherence', color='blue')
    ax3_twin.set_ylabel('Mean Phase (rad)', color='red')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')

    ax3.set_ylim(0, 1.1)
    ax3_twin.set_ylim(-np.pi, np.pi)

    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='upper right', fontsize=9)

    ax3.set_title('Phase Coherence and Mean Phase\n(Quantum Decoherence Analysis)',
                  fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # ===================== Chart 4: Spatial Localization Metrics (Bottom-Right) =====================
    ax4 = fig.add_subplot(2, 2, 4)

    # Compute localization metrics over time
    # 1. Position expectation values <x>, <y>, <z>
    # 2. Position uncertainties Δx, Δy, Δz
    # 3. Participation ratio (inverse of IPR)

    pos_x = np.zeros(n_times)
    pos_y = np.zeros(n_times)
    pos_z = np.zeros(n_times)
    delta_x = np.zeros(n_times)
    delta_y = np.zeros(n_times)
    delta_z = np.zeros(n_times)
    ipr = np.zeros(n_times)  # Inverse participation ratio

    X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')

    for t in range(n_times):
        prob_t = prob_density[t]
        norm = np.sum(prob_t)

        if norm > 0:
            prob_norm = prob_t / norm

            # Expectation values
            pos_x[t] = np.sum(prob_norm * X3)
            pos_y[t] = np.sum(prob_norm * Y3)
            pos_z[t] = np.sum(prob_norm * Z3)

            # Variances
            delta_x[t] = np.sqrt(np.sum(prob_norm * (X3 - pos_x[t])**2))
            delta_y[t] = np.sqrt(np.sum(prob_norm * (Y3 - pos_y[t])**2))
            delta_z[t] = np.sqrt(np.sum(prob_norm * (Z3 - pos_z[t])**2))

            # IPR: sum of prob^2 (higher = more localized)
            ipr[t] = np.sum(prob_norm**2)

    # Plot position uncertainties
    ax4.plot(times_fs[:n_times], delta_x, 'r-', linewidth=2, marker='o',
             markersize=5, label='Δx (Å)')
    ax4.plot(times_fs[:n_times], delta_y, 'g-', linewidth=2, marker='s',
             markersize=5, label='Δy (Å)')
    ax4.plot(times_fs[:n_times], delta_z, 'b-', linewidth=2, marker='^',
             markersize=5, label='Δz (Å)')

    # Total uncertainty
    delta_total = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    ax4.plot(times_fs[:n_times], delta_total, 'k-', linewidth=3,
             label='Δr total (Å)')

    # Add IPR on secondary axis
    ax4_twin = ax4.twinx()
    ax4_twin.plot(times_fs[:n_times], ipr * 1e6, 'm--', linewidth=2,
                  label='IPR × 10⁶')
    ax4_twin.set_ylabel('IPR × 10⁶', color='magenta')
    ax4_twin.tick_params(axis='y', labelcolor='magenta')

    ax4.set_xlabel('Time (fs)')
    ax4.set_ylabel('Position Uncertainty (Å)')
    ax4.legend(loc='upper left', fontsize=9)
    ax4_twin.legend(loc='upper right', fontsize=9)

    ax4.set_title('Spatial Localization Metrics\n(Wavepacket Width & IPR)',
                  fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # Main title
    transfer_time = results.get('experiment', {}).get('transfer_time_fs', 850)
    fig.suptitle(f'Panel 7: Wavefunction Analysis\n'
                 f'Electron Transfer Dynamics | Grid: {nx}×{ny}×{nz} | {n_times} timesteps',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 8: 3D Electron Cloud Panel (4 3D charts)
# =============================================================================

def plot_panel_8_electron_cloud(wavefunction: Dict[str, np.ndarray], results: Dict[str, Any],
                                 output_path: Path) -> plt.Figure:
    """
    Panel 8: 3D Electron Probability Cloud Visualization.

    Layout (2x2) - ALL 3D CHARTS:
    - Top-left: Electron cloud at t=0 (initial state at Cu donor)
    - Top-right: Electron cloud at t=mid (during transfer)
    - Bottom-left: Electron cloud at t=end (final state)
    - Bottom-right: Time-lapse overlay (multiple timesteps combined)
    """
    fig = plt.figure(figsize=(18, 16))

    psi_real = wavefunction['psi_real']
    psi_imag = wavefunction['psi_imag']
    grid_min = wavefunction['grid_min']
    grid_max = wavefunction['grid_max']

    n_times, nx, ny, nz = psi_real.shape

    # Compute probability density |ψ|²
    prob_density = psi_real**2 + psi_imag**2

    # Create coordinate arrays (in Angstroms)
    x = np.linspace(grid_min[0], grid_max[0], nx) * 1e10
    y = np.linspace(grid_min[1], grid_max[1], ny) * 1e10
    z = np.linspace(grid_min[2], grid_max[2], nz) * 1e10

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Get time array
    trajectory = results.get('trajectory', [])
    if trajectory:
        times_fs = np.array([step['time'] for step in trajectory[:n_times]]) * 1e15
    else:
        times_fs = np.arange(n_times) * 10.0

    # Select 4 key timesteps
    t_indices = [0, n_times // 3, 2 * n_times // 3, n_times - 1]
    titles = ['Initial State (t = 0 fs)',
              f'Early Transfer (t = {times_fs[t_indices[1]]:.0f} fs)',
              f'Late Transfer (t = {times_fs[t_indices[2]]:.0f} fs)',
              f'Final State (t = {times_fs[t_indices[3]]:.0f} fs)']

    # Copper ligand positions for reference
    ligands = {
        'Cu': np.array([0.0, 0.0, 0.0]),
        'His46': np.array([2.0, 0.0, 0.0]),
        'Cys112': np.array([0.0, 2.1, 0.0]),
        'His117': np.array([-2.0, 0.0, 0.0]),
        'Met121': np.array([0.0, 0.0, 3.1])
    }

    def plot_electron_cloud(ax, prob_t, time_label, cmap_name='Blues'):
        """Helper function to plot 3D electron probability cloud."""
        # Find threshold for isosurface (adaptive based on data)
        max_prob = np.max(prob_t)
        if max_prob < 1e-10:
            ax.text(0, 0, 0, 'Very low\nprobability', ha='center', fontsize=12)
            return

        # Multiple isosurface levels for cloud effect
        thresholds = [0.1 * max_prob, 0.3 * max_prob, 0.6 * max_prob]
        alphas = [0.2, 0.4, 0.7]
        sizes = [15, 25, 40]

        cmap = plt.cm.get_cmap(cmap_name)

        for thresh, alpha, size in zip(thresholds, alphas, sizes):
            mask = prob_t > thresh
            if not np.any(mask):
                continue

            # Get points above threshold
            indices = np.where(mask)
            n_points = len(indices[0])

            # Downsample if too many points
            max_points = 1500
            if n_points > max_points:
                sample_idx = np.random.choice(n_points, max_points, replace=False)
                px = X[mask][sample_idx]
                py = Y[mask][sample_idx]
                pz = Z[mask][sample_idx]
                pprob = prob_t[mask][sample_idx]
            else:
                px = X[mask]
                py = Y[mask]
                pz = Z[mask]
                pprob = prob_t[mask]

            # Normalize probability for coloring
            pprob_norm = pprob / max_prob

            # Plot as scatter with color based on probability
            ax.scatter(px, py, pz, c=pprob_norm, cmap=cmap_name,
                      s=size, alpha=alpha, edgecolors='none',
                      vmin=0, vmax=1)

        # Add copper center marker
        ax.scatter(0, 0, 0, s=300, c='orange', marker='o',
                  edgecolors='black', linewidths=2, label='Cu', zorder=10)

        # Add ligand positions
        for name, pos in ligands.items():
            if name != 'Cu':
                ax.scatter(*pos, s=80, c='purple', marker='s',
                          edgecolors='black', linewidths=0.5, alpha=0.7)
                # Bond to copper
                ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]],
                       'purple', linewidth=1, alpha=0.4)

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(time_label, fontweight='bold', fontsize=12)

        # Set consistent axis limits
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())

    # ===================== Chart 1: Initial State (Top-Left) =====================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_electron_cloud(ax1, prob_density[t_indices[0]], titles[0], 'Blues')
    ax1.view_init(elev=20, azim=-60)

    # ===================== Chart 2: Early Transfer (Top-Right) =====================
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_electron_cloud(ax2, prob_density[t_indices[1]], titles[1], 'Greens')
    ax2.view_init(elev=20, azim=-120)

    # ===================== Chart 3: Late Transfer (Bottom-Left) =====================
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_electron_cloud(ax3, prob_density[t_indices[2]], titles[2], 'Oranges')
    ax3.view_init(elev=20, azim=60)

    # ===================== Chart 4: Time-Lapse Overlay (Bottom-Right) =====================
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    # Overlay multiple timesteps with different colors
    time_colors = ['blue', 'green', 'orange', 'red']
    time_labels_short = ['t=0', f't={times_fs[t_indices[1]]:.0f}fs',
                         f't={times_fs[t_indices[2]]:.0f}fs', f't={times_fs[-1]:.0f}fs']

    for idx, (t_idx, color, label) in enumerate(zip(t_indices, time_colors, time_labels_short)):
        prob_t = prob_density[t_idx]
        max_prob = np.max(prob_t)

        if max_prob < 1e-10:
            continue

        # Use 30% threshold for overlay
        threshold = 0.3 * max_prob
        mask = prob_t > threshold

        if not np.any(mask):
            continue

        indices = np.where(mask)
        n_points = len(indices[0])

        # Downsample
        max_points = 800
        if n_points > max_points:
            sample_idx = np.random.choice(n_points, max_points, replace=False)
            px = X[mask][sample_idx]
            py = Y[mask][sample_idx]
            pz = Z[mask][sample_idx]
        else:
            px = X[mask]
            py = Y[mask]
            pz = Z[mask]

        ax4.scatter(px, py, pz, c=color, s=20, alpha=0.4,
                   edgecolors='none', label=label)

    # Copper center
    ax4.scatter(0, 0, 0, s=400, c='orange', marker='o',
               edgecolors='black', linewidths=2, label='Cu center', zorder=10)

    # Ligands
    for name, pos in ligands.items():
        if name != 'Cu':
            ax4.scatter(*pos, s=100, c='purple', marker='s',
                       edgecolors='black', linewidths=1, alpha=0.8)
            ax4.plot([0, pos[0]], [0, pos[1]], [0, pos[2]],
                    'purple', linewidth=1.5, alpha=0.5)

    ax4.set_xlabel('X (Å)')
    ax4.set_ylabel('Y (Å)')
    ax4.set_zlabel('Z (Å)')
    ax4.set_title('Time-Lapse Overlay\n(All States Combined)', fontweight='bold', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.view_init(elev=30, azim=-45)

    ax4.set_xlim(x.min(), x.max())
    ax4.set_ylim(y.min(), y.max())
    ax4.set_zlim(z.min(), z.max())

    # Add colorbar legend
    sm = ScalarMappable(cmap='viridis', norm=Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('|ψ|² (normalized)', fontsize=11)

    # Main title
    transfer_time = results.get('experiment', {}).get('transfer_time_fs', 850)
    fig.suptitle(f'Panel 8: 3D Electron Probability Cloud\n'
                 f'Cu(I) → Cu(II) Electron Transfer | τ = {transfer_time:.0f} fs | '
                 f'Grid: {nx}×{ny}×{nz}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 9: Perturbation Fields (2 3D + 2 metrics)
# =============================================================================

def plot_panel_9_perturbation_fields(results: Dict, output_path: Path) -> plt.Figure:
    """
    Panel 9: Perturbation Field Visualization (2 3D + 2 metrics)

    Charts:
    1. 3D Electric Field (P1 - radial perturbation)
    2. 3D Magnetic Field (P2 - angular perturbation)
    3. Field Magnitude Evolution
    4. Perturbation Response Summary
    """
    fig = plt.figure(figsize=(20, 16))

    # Extract trajectory data
    trajectory = results['trajectory']
    n_points = len(trajectory)

    # Azurin ligand positions (in Angstroms)
    cu_center = np.array([0, 0, 0])
    ligands = {
        'His46': np.array([2.1, 0.5, 1.8]),
        'Cys112': np.array([-1.5, 2.0, 0.8]),
        'His117': np.array([0.8, -1.8, 1.5]),
        'Met121': np.array([-0.5, 0.3, -2.5])
    }

    # Generate grid for field visualization
    grid_size = 15
    x = np.linspace(-8, 8, grid_size)
    y = np.linspace(-8, 8, grid_size)
    z = np.linspace(-8, 8, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    # Compute electric field (P1 - radial perturbation)
    # E-field from Cu center and electron position
    electron_positions = np.array([t['position'] for t in trajectory]) * 1e10  # Convert to Angstroms

    # Use midpoint of trajectory for field visualization
    mid_idx = n_points // 2
    e_pos = electron_positions[mid_idx]

    # Electric field from point charges (Cu²⁺ and electron)
    def compute_E_field(X, Y, Z, charge_pos, charge):
        """Compute electric field from point charge."""
        k_e = 8.99e9  # Coulomb constant
        eps = 1e-10

        dx = X - charge_pos[0]
        dy = Y - charge_pos[1]
        dz = Z - charge_pos[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2 + eps)

        E_mag = k_e * charge / (r**2 + eps)
        Ex = E_mag * dx / r
        Ey = E_mag * dy / r
        Ez = E_mag * dz / r

        return Ex, Ey, Ez

    # Cu²⁺ field + electron field
    Ex_cu, Ey_cu, Ez_cu = compute_E_field(X, Y, Z, cu_center, 2)  # Cu²⁺
    Ex_e, Ey_e, Ez_e = compute_E_field(X, Y, Z, e_pos, -1)  # electron

    Ex_total = Ex_cu + Ex_e
    Ey_total = Ey_cu + Ey_e
    Ez_total = Ez_cu + Ez_e
    E_mag = np.sqrt(Ex_total**2 + Ey_total**2 + Ez_total**2)

    # Compute magnetic field (P2 - angular perturbation)
    # Simplified B-field from electron motion
    def compute_B_field(X, Y, Z, e_pos, velocity):
        """Compute magnetic field from moving electron."""
        mu_0 = 4 * np.pi * 1e-7
        q = 1.6e-19
        eps = 1e-10

        dx = X - e_pos[0]
        dy = Y - e_pos[1]
        dz = Z - e_pos[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2 + eps)

        # B = (μ₀/4π) * q(v × r̂)/r²
        # Cross product: v × r
        vx, vy, vz = velocity
        Bx = mu_0 * q / (4 * np.pi) * (vy * dz - vz * dy) / (r**3 + eps)
        By = mu_0 * q / (4 * np.pi) * (vz * dx - vx * dz) / (r**3 + eps)
        Bz = mu_0 * q / (4 * np.pi) * (vx * dy - vy * dx) / (r**3 + eps)

        return Bx, By, Bz

    # Estimate velocity from trajectory
    if mid_idx > 0:
        velocity = (electron_positions[mid_idx] - electron_positions[mid_idx-1]) / 1e-15  # Å/fs
    else:
        velocity = np.array([0.5, 0.3, 0.2])  # Default

    Bx, By, Bz = compute_B_field(X, Y, Z, e_pos, velocity * 1e5)  # Scale for visualization
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # -------------------------------------------------------------------------
    # Chart 1: 3D Electric Field (P1)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Sample points for quiver
    step = 3
    xs = X[::step, ::step, ::step].flatten()
    ys = Y[::step, ::step, ::step].flatten()
    zs = Z[::step, ::step, ::step].flatten()
    Ex_s = Ex_total[::step, ::step, ::step].flatten()
    Ey_s = Ey_total[::step, ::step, ::step].flatten()
    Ez_s = Ez_total[::step, ::step, ::step].flatten()

    # Normalize arrows
    E_norm = np.sqrt(Ex_s**2 + Ey_s**2 + Ez_s**2)
    E_norm[E_norm == 0] = 1
    scale = 2.0

    ax1.quiver(xs, ys, zs,
               Ex_s/E_norm*scale, Ey_s/E_norm*scale, Ez_s/E_norm*scale,
               color='red', alpha=0.6, arrow_length_ratio=0.3, linewidth=0.8)

    # Cu center
    ax1.scatter(0, 0, 0, s=300, c='orange', marker='o', edgecolors='black',
                linewidths=2, label='Cu²⁺', zorder=10)

    # Electron position
    ax1.scatter(*e_pos, s=150, c='blue', marker='o', edgecolors='black',
                linewidths=1, label=f'e⁻ (t={mid_idx})', zorder=10)

    # Ligands
    for name, pos in ligands.items():
        ax1.scatter(*pos, s=80, c='green', marker='s', alpha=0.7)
        ax1.text(pos[0], pos[1], pos[2]+0.5, name, fontsize=8, ha='center')

    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title('P1: Electric Field (Radial Perturbation)\n∇ᵣρ → Linear Momentum Coupling',
                  fontweight='bold', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.view_init(elev=25, azim=-60)

    # -------------------------------------------------------------------------
    # Chart 2: 3D Magnetic Field (P2)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')

    Bx_s = Bx[::step, ::step, ::step].flatten()
    By_s = By[::step, ::step, ::step].flatten()
    Bz_s = Bz[::step, ::step, ::step].flatten()

    # Normalize arrows
    B_norm = np.sqrt(Bx_s**2 + By_s**2 + Bz_s**2)
    B_norm[B_norm == 0] = 1

    ax2.quiver(xs, ys, zs,
               Bx_s/B_norm*scale, By_s/B_norm*scale, Bz_s/B_norm*scale,
               color='purple', alpha=0.6, arrow_length_ratio=0.3, linewidth=0.8)

    # Cu center
    ax2.scatter(0, 0, 0, s=300, c='orange', marker='o', edgecolors='black',
                linewidths=2, label='Cu²⁺', zorder=10)

    # Electron trajectory arc
    ax2.plot(electron_positions[:, 0], electron_positions[:, 1], electron_positions[:, 2],
             'b-', linewidth=2, alpha=0.7, label='e⁻ path')
    ax2.scatter(*e_pos, s=150, c='blue', marker='o', edgecolors='black',
                linewidths=1, zorder=10)

    # Velocity vector
    v_scale = 0.5
    ax2.quiver(e_pos[0], e_pos[1], e_pos[2],
               velocity[0]*v_scale, velocity[1]*v_scale, velocity[2]*v_scale,
               color='cyan', linewidth=3, arrow_length_ratio=0.2, label='velocity')

    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_zlabel('Z (Å)')
    ax2.set_title('P2: Magnetic Field (Angular Perturbation)\n∇_θρ → Angular Momentum Coupling',
                  fontweight='bold', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.view_init(elev=25, azim=-60)

    # -------------------------------------------------------------------------
    # Chart 3: Field Magnitude Evolution
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)

    # Compute field magnitudes at each trajectory point
    E_mags = []
    B_mags = []
    times = []

    for i, t_data in enumerate(trajectory):
        pos = np.array(t_data['position']) * 1e10

        # Distance from Cu
        r_cu = np.linalg.norm(pos - cu_center)

        # E-field magnitude at electron position (simplified)
        E_at_e = 8.99e9 * 2 / (r_cu**2 + 0.1)  # From Cu²⁺
        E_mags.append(E_at_e / 1e10)  # Normalize

        # B-field estimate
        if i > 0:
            v = (electron_positions[i] - electron_positions[i-1]) / 50  # fs timestep
            v_mag = np.linalg.norm(v)
        else:
            v_mag = 1.0
        B_at_e = 4e-7 * 1.6e-19 * v_mag / (r_cu**2 + 0.1)
        B_mags.append(B_at_e * 1e15)  # Normalize

        times.append(t_data.get('time', i * 50) * 1e15)  # fs

    ax3.plot(times, E_mags, 'r-', linewidth=2, label='|E| (P1 radial)', marker='o', markersize=4)
    ax3.plot(times, B_mags, 'purple', linewidth=2, label='|B| (P2 angular)', marker='s', markersize=4)

    ax3.set_xlabel('Time (fs)', fontsize=11)
    ax3.set_ylabel('Field Magnitude (arb. units)', fontsize=11)
    ax3.set_title('Perturbation Field Evolution During Transfer', fontweight='bold', fontsize=11)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(times))

    # Mark key transitions
    for i, t in enumerate(times):
        if i < len(trajectory) and 'trit' in trajectory[i]:
            trit = trajectory[i]['trit']
            if trit == 2:  # Null response
                ax3.axvline(t, color='gray', alpha=0.3, linestyle='--')

    # -------------------------------------------------------------------------
    # Chart 4: Perturbation Response Summary
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)

    # Count responses
    trits = [t.get('trit', 1) for t in trajectory]
    response_counts = {
        'Radial (trit=0)': trits.count(0),
        'Angular (trit=1)': trits.count(1),
        'Null (trit=2)': trits.count(2)
    }

    colors = ['red', 'purple', 'gray']
    bars = ax4.bar(response_counts.keys(), response_counts.values(), color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, response_counts.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Perturbation Response Classification\n[P1, P2] Response Encoding',
                  fontweight='bold', fontsize=11)
    ax4.set_ylim(0, max(response_counts.values()) * 1.2)

    # Add encoding legend
    encoding_text = (
        "Encoding:\n"
        "• (1,0) → trit=0: Radial response\n"
        "• (0,1) → trit=1: Angular response\n"
        "• (0,0) → trit=2: Null response"
    )
    ax4.text(0.95, 0.95, encoding_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add summary stats
    total = len(trits)
    ternary_string = ''.join(str(t) for t in trits)
    stats_text = (
        f"Total iterations: {total}\n"
        f"Ternary string: {ternary_string}\n"
        f"Information bits: {total * np.log2(3):.1f}"
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Main title
    fig.suptitle('Panel 9: Perturbation Field Analysis\n'
                 'P1 (Electric/Radial) & P2 (Magnetic/Angular) Internal Perturbations',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Panel 10: Ternary Trisection Method (1 3D + 3 charts)
# =============================================================================

def plot_panel_10_ternary_trisection(results: Dict, output_path: Path) -> plt.Figure:
    """
    Panel 10: Ternary Trisection Method Visualization (1 3D + 3 charts)

    Charts:
    1. 3D Trisection Search Space (showing subdivision)
    2. Convergence Analysis
    3. Complexity Comparison (O(log₃ N) vs O(N))
    4. Localization Precision vs Iteration
    """
    fig = plt.figure(figsize=(20, 16))

    # Extract data
    trajectory = results['trajectory']
    n_iterations = len(trajectory)

    # Initial search region (Azurin active site ~20 Å)
    initial_min = np.array([-10, -10, -10])  # Angstroms
    initial_max = np.array([10, 10, 10])

    # -------------------------------------------------------------------------
    # Chart 1: 3D Trisection Search Space
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Draw initial bounding box
    def draw_box(ax, min_pt, max_pt, color, alpha=0.3, linewidth=1):
        """Draw a 3D box wireframe."""
        x = [min_pt[0], max_pt[0]]
        y = [min_pt[1], max_pt[1]]
        z = [min_pt[2], max_pt[2]]

        # Draw edges
        for i in range(2):
            for j in range(2):
                ax.plot([x[0], x[1]], [y[i], y[i]], [z[j], z[j]], color=color, alpha=alpha, linewidth=linewidth)
                ax.plot([x[i], x[i]], [y[0], y[1]], [z[j], z[j]], color=color, alpha=alpha, linewidth=linewidth)
                ax.plot([x[i], x[i]], [y[j], y[j]], [z[0], z[1]], color=color, alpha=alpha, linewidth=linewidth)

    # Draw initial region
    draw_box(ax1, initial_min, initial_max, 'blue', alpha=0.2, linewidth=2)

    # Simulate trisection convergence
    region_min = initial_min.copy()
    region_max = initial_max.copy()

    trits = [t.get('trit', 1) for t in trajectory]
    electron_positions = np.array([t['position'] for t in trajectory]) * 1e10
    final_pos = electron_positions[-1]

    colors = plt.cm.viridis(np.linspace(0, 1, min(6, n_iterations)))

    for i in range(min(6, n_iterations)):  # Show first 6 subdivisions
        axis = i % 3
        trit = trits[i]

        region_size = region_max - region_min

        if trit == 0:
            region_max[axis] = region_min[axis] + region_size[axis] / 3
        elif trit == 1:
            new_min = region_min[axis] + region_size[axis] / 3
            new_max = region_min[axis] + 2 * region_size[axis] / 3
            region_min[axis] = new_min
            region_max[axis] = new_max
        else:  # trit == 2
            region_min[axis] = region_min[axis] + 2 * region_size[axis] / 3

        draw_box(ax1, region_min.copy(), region_max.copy(), colors[i], alpha=0.4, linewidth=1.5)

    # Final localized region
    draw_box(ax1, region_min, region_max, 'red', alpha=0.8, linewidth=3)

    # Electron trajectory
    ax1.plot(electron_positions[:, 0], electron_positions[:, 1], electron_positions[:, 2],
             'g-', linewidth=2, alpha=0.8, label='Electron path')
    ax1.scatter(*final_pos, s=200, c='red', marker='*', edgecolors='black',
                linewidths=1, label='Final position', zorder=10)

    # Cu center
    ax1.scatter(0, 0, 0, s=300, c='orange', marker='o', edgecolors='black',
                linewidths=2, label='Cu center', zorder=10)

    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title('3D Ternary Trisection Search Space\nProgressive Region Subdivision',
                  fontweight='bold', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.view_init(elev=20, azim=-45)

    # -------------------------------------------------------------------------
    # Chart 2: Convergence Analysis
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)

    # Compute region volume at each iteration
    volumes = []
    uncertainties = []
    region_min = initial_min.copy()
    region_max = initial_max.copy()

    initial_volume = np.prod(initial_max - initial_min)
    volumes.append(initial_volume)
    uncertainties.append(np.linalg.norm(initial_max - initial_min) / np.sqrt(3))

    for i in range(n_iterations):
        axis = i % 3
        trit = trits[i]
        region_size = region_max - region_min

        if trit == 0:
            region_max[axis] = region_min[axis] + region_size[axis] / 3
        elif trit == 1:
            new_min = region_min[axis] + region_size[axis] / 3
            new_max = region_min[axis] + 2 * region_size[axis] / 3
            region_min[axis] = new_min
            region_max[axis] = new_max
        else:
            region_min[axis] = region_min[axis] + 2 * region_size[axis] / 3

        vol = np.prod(region_max - region_min)
        volumes.append(vol)
        uncertainties.append(np.linalg.norm(region_max - region_min) / np.sqrt(3))

    iterations = range(n_iterations + 1)

    ax2.semilogy(iterations, volumes, 'b-o', linewidth=2, markersize=6, label='Search volume (Å³)')
    ax2.semilogy(iterations, uncertainties, 'r-s', linewidth=2, markersize=6, label='Position uncertainty (Å)')

    # Theoretical curves
    theory_vol = [initial_volume / (3**i) for i in iterations]
    ax2.semilogy(iterations, theory_vol, 'b--', alpha=0.5, label='Theory: V/3ᵏ')

    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Volume / Uncertainty', fontsize=11)
    ax2.set_title('Trisection Convergence\nExponential Volume Reduction', fontweight='bold', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_iterations)

    # -------------------------------------------------------------------------
    # Chart 3: Complexity Comparison
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)

    # Compare O(log₃ N) vs O(N) vs O(log₂ N)
    N_values = np.logspace(1, 6, 50)

    ternary_complexity = np.log(N_values) / np.log(3)
    binary_complexity = np.log(N_values) / np.log(2)
    linear_complexity = N_values

    ax3.loglog(N_values, linear_complexity, 'r-', linewidth=2, label='O(N) - Linear search')
    ax3.loglog(N_values, binary_complexity, 'g-', linewidth=2, label='O(log₂ N) - Binary')
    ax3.loglog(N_values, ternary_complexity, 'b-', linewidth=3, label='O(log₃ N) - Ternary trisection')

    # Mark actual experiment
    N_actual = 3**n_iterations
    ax3.axvline(N_actual, color='purple', linestyle='--', alpha=0.7, label=f'This exp: N={N_actual:.0e}')
    ax3.scatter([N_actual], [n_iterations], s=200, c='purple', marker='*', zorder=10)

    ax3.set_xlabel('Search Space Size (N grid points)', fontsize=11)
    ax3.set_ylabel('Number of Measurements', fontsize=11)
    ax3.set_title('Complexity Comparison\nTernary Trisection Advantage', fontweight='bold', fontsize=11)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')

    # Add speedup annotation
    speedup_vs_linear = N_actual / n_iterations
    speedup_vs_binary = (np.log(N_actual) / np.log(2)) / n_iterations
    ax3.text(0.95, 0.5, f'Speedup vs linear: {speedup_vs_linear:.1e}×\n'
                        f'Speedup vs binary: {speedup_vs_binary:.2f}×',
             transform=ax3.transAxes, fontsize=10, verticalalignment='center',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # -------------------------------------------------------------------------
    # Chart 4: Localization Precision vs Iteration
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)

    # Precision in each dimension
    precisions = []
    region_min = initial_min.copy()
    region_max = initial_max.copy()

    for i in range(n_iterations):
        axis = i % 3
        trit = trits[i]
        region_size = region_max - region_min

        if trit == 0:
            region_max[axis] = region_min[axis] + region_size[axis] / 3
        elif trit == 1:
            new_min = region_min[axis] + region_size[axis] / 3
            new_max = region_min[axis] + 2 * region_size[axis] / 3
            region_min[axis] = new_min
            region_max[axis] = new_max
        else:
            region_min[axis] = region_min[axis] + 2 * region_size[axis] / 3

        precisions.append({
            'x': region_max[0] - region_min[0],
            'y': region_max[1] - region_min[1],
            'z': region_max[2] - region_min[2]
        })

    iters = range(1, n_iterations + 1)
    ax4.semilogy(iters, [p['x'] for p in precisions], 'r-o', linewidth=2, markersize=5, label='Δx')
    ax4.semilogy(iters, [p['y'] for p in precisions], 'g-s', linewidth=2, markersize=5, label='Δy')
    ax4.semilogy(iters, [p['z'] for p in precisions], 'b-^', linewidth=2, markersize=5, label='Δz')

    # Combined uncertainty
    total_uncertainty = [np.sqrt(p['x']**2 + p['y']**2 + p['z']**2) for p in precisions]
    ax4.semilogy(iters, total_uncertainty, 'k-', linewidth=3, label='|Δr| total')

    # Target precision
    target = 0.1  # 0.1 Å
    ax4.axhline(target, color='purple', linestyle='--', linewidth=2, label=f'Target: {target} Å')

    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Position Uncertainty (Å)', fontsize=11)
    ax4.set_title('Localization Precision per Dimension\nCyclic Axis Refinement', fontweight='bold', fontsize=11)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, n_iterations)

    # Final precision annotation
    final_precision = total_uncertainty[-1] if total_uncertainty else 1.0
    ax4.text(0.05, 0.05, f'Final precision: {final_precision:.3f} Å\n'
                         f'Iterations: {n_iterations}\n'
                         f'Ternary string: {"".join(str(t) for t in trits)}',
             transform=ax4.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Main title
    fig.suptitle('Panel 10: Ternary Trisection Localization Method\n'
                 f'O(log₃ N) Zero-Backaction Electron Localization | {n_iterations} iterations | '
                 f'Final Δr = {final_precision:.3f} Å',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig


# =============================================================================
# Main Function
# =============================================================================

def generate_all_panels(data_dir: Path = None, output_dir: Path = None) -> None:
    """Generate all 8 visualization panels (4 charts each, 32 charts total)."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data' / 'processed'

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'visualizations'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING MULTI-CHART VISUALIZATION PANELS")
    print("=" * 80)

    # Load results
    results_path = data_dir / 'validation_results.json'
    wavefunction_path = data_dir / 'wavefunction.npz'

    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return

    print(f"\n[1/7] Loading results from {results_path}...")
    results = load_results(results_path)

    # Load wavefunction
    wavefunction = None
    if wavefunction_path.exists():
        print(f"[2/7] Loading wavefunction from {wavefunction_path}...")
        wavefunction = load_wavefunction(wavefunction_path)
    else:
        print("[2/7] Creating synthetic wavefunction data...")
        grid_min = np.array([-20, -20, -20]) * 1e-10
        grid_max = np.array([20, 20, 20]) * 1e-10
        n_times = len(results['trajectory'])
        shape = (n_times, 40, 40, 40)
        wavefunction = {
            'psi_real': np.random.randn(*shape) * 0.1,
            'psi_imag': np.random.randn(*shape) * 0.1,
            'grid_min': grid_min,
            'grid_max': grid_max,
            'resolution': 1e-10
        }

    # Generate panels
    print("\n[3/7] Panel 1: 3D Trajectory (4 charts)...")
    plot_panel_1_trajectory(results, output_dir / 'panel_1_trajectory.png')
    print("      -> panel_1_trajectory.png")

    print("\n[4/7] Panel 2: Backaction Verification (4 charts)...")
    plot_panel_2_backaction(results, output_dir / 'panel_2_backaction.png')
    print("      -> panel_2_backaction.png")

    print("\n[5/7] Panel 3: Categorical Coordinates (4 charts)...")
    plot_panel_3_categorical(results, output_dir / 'panel_3_categorical.png')
    print("      -> panel_3_categorical.png")

    print("\n[6/7] Panel 4: Probability Density (4 charts)...")
    plot_panel_4_probability(results, wavefunction, output_dir / 'panel_4_probability.png')
    print("      -> panel_4_probability.png")

    print("\n[7/7] Panel 5: S-Entropy Space (4 charts)...")
    plot_panel_5_sentropy(results, output_dir / 'panel_5_sentropy.png')
    print("      -> panel_5_sentropy.png")

    print("\n[8/7] Panel 6: Protein Structure (4 3D charts)...")
    plot_panel_6_protein_structure(results, output_dir / 'panel_6_protein_structure.png')
    print("      -> panel_6_protein_structure.png")

    print("\n[9/10] Panel 7: Wavefunction Analysis (4 charts)...")
    plot_panel_7_wavefunction(wavefunction, results, output_dir / 'panel_7_wavefunction.png')
    print("      -> panel_7_wavefunction.png")

    print("\n[10/12] Panel 8: 3D Electron Cloud (4 3D charts)...")
    plot_panel_8_electron_cloud(wavefunction, results, output_dir / 'panel_8_electron_cloud.png')
    print("      -> panel_8_electron_cloud.png")

    print("\n[11/12] Panel 9: Perturbation Fields (2 3D + 2 metrics)...")
    plot_panel_9_perturbation_fields(results, output_dir / 'panel_9_perturbation_fields.png')
    print("      -> panel_9_perturbation_fields.png")

    print("\n[12/12] Panel 10: Ternary Trisection Method (1 3D + 3 charts)...")
    plot_panel_10_ternary_trisection(results, output_dir / 'panel_10_ternary_trisection.png')
    print("      -> panel_10_ternary_trisection.png")

    print("\n" + "=" * 80)
    print("ALL 10 PANELS GENERATED (40 CHARTS TOTAL)")
    print("=" * 80)
    print(f"\nOutput: {output_dir}")
    print("\nFiles:")
    for i in range(1, 11):
        print(f"  - panel_{i}_*.png (4 charts)")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_panels()
