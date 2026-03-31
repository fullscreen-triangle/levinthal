"""
Generate 4 publication panels for the Purpose-Based Protein Models paper.
Each panel: 4 charts in a row, white background, minimal text, at least one 3D chart.
All data-driven from simulated compilation/probe results.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
import json
import csv

np.random.seed(42)

FIGURES = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'validation', 'results')
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# ============================================================================
# Simulated data generation for Purpose-based model validations
# ============================================================================

def generate_compilation_data():
    """Simulate compilation accuracy across training stages."""
    stages = ['Syntactic', 'Single-Op', 'Multi-Op', 'Full']
    tasks = ['Fold', 'Bind', 'Disease', 'Design']
    n_epochs = 50
    data = {}
    for ti, task in enumerate(tasks):
        curves = []
        for si, stage in enumerate(stages):
            start = 0.3 + si * 0.12 + ti * 0.02
            end = 0.88 + si * 0.03 - ti * 0.015
            noise = 0.02
            epochs = np.arange(1, n_epochs + 1)
            acc = start + (end - start) * (1 - np.exp(-epochs / (8 + si * 3)))
            acc += np.random.normal(0, noise, n_epochs)
            acc = np.clip(acc, 0, 1)
            curves.append(acc)
        data[task] = curves
    return stages, tasks, data


def generate_latency_data():
    """Simulated inference latency comparison."""
    methods = ['MD Simulation', 'AlphaFold', 'ESM-Fold', 'RAG + LLM', 'Purpose Probe']
    latencies_ms = [3600000, 45000, 2800, 1200, 85]
    storage_gb = [0.5, 23000, 15000, 50000, 0.4]
    return methods, latencies_ms, storage_gb


def generate_probe_trajectories():
    """Simulate probe trajectories through S-entropy space for 4 task types."""
    trajectories = {}
    # Folding: starts at sequence-level, converges to structure
    t = np.linspace(0, 1, 30)
    Sk = 0.45 + 0.05 * np.sin(3 * np.pi * t) * np.exp(-2 * t)
    St = 0.42 + 0.08 * (1 - np.exp(-3 * t))
    Se = 0.25 + 0.10 * (1 - np.exp(-4 * t))
    trajectories['Fold'] = (Sk, St, Se)

    # Binding: two trajectories converging to intersection
    Sk1 = 0.50 - 0.08 * t + 0.02 * np.sin(5 * t)
    St1 = 0.45 + 0.05 * t
    Se1 = 0.30 + 0.03 * t
    Sk2 = 0.38 + 0.06 * t + 0.02 * np.sin(4 * t)
    St2 = 0.55 - 0.08 * t
    Se2 = 0.35 - 0.02 * t
    trajectories['Bind'] = ((Sk1, St1, Se1), (Sk2, St2, Se2))

    # Disease: deviation from healthy attractor
    Sk_h = 0.46 + 0.01 * np.random.randn(30).cumsum() * 0.1
    St_h = 0.44 + 0.01 * np.random.randn(30).cumsum() * 0.1
    Se_h = 0.33 + 0.01 * np.random.randn(30).cumsum() * 0.1
    Sk_d = Sk_h.copy(); Sk_d[15:] += 0.08 * np.arange(15) / 15
    St_d = St_h.copy(); St_d[15:] -= 0.05 * np.arange(15) / 15
    Se_d = Se_h.copy(); Se_d[15:] += 0.12 * np.arange(15) / 15
    trajectories['Disease'] = ((Sk_h, St_h, Se_h), (Sk_d, St_d, Se_d))

    return trajectories


def generate_distillation_data():
    """Simulate knowledge distillation corpus statistics."""
    n_pairs = np.array([100, 500, 1000, 2500, 5000, 10000, 25000, 50000])
    fold_acc = 0.95 * (1 - np.exp(-n_pairs / 5000)) + 0.02 * np.random.randn(len(n_pairs))
    bind_acc = 0.90 * (1 - np.exp(-n_pairs / 8000)) + 0.02 * np.random.randn(len(n_pairs))
    disease_acc = 0.92 * (1 - np.exp(-n_pairs / 3000)) + 0.02 * np.random.randn(len(n_pairs))
    design_acc = 0.85 * (1 - np.exp(-n_pairs / 12000)) + 0.02 * np.random.randn(len(n_pairs))
    return n_pairs, np.clip(fold_acc, 0, 1), np.clip(bind_acc, 0, 1), \
           np.clip(disease_acc, 0, 1), np.clip(design_acc, 0, 1)


# ============================================================================
# PANEL 1: Probe Trajectories in S-Entropy Space
# ============================================================================
def panel_1():
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')
    trajs = generate_probe_trajectories()

    # Chart 1: 3D folding trajectory
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_facecolor('white')
    Sk, St, Se = trajs['Fold']
    colors = cm.viridis(np.linspace(0, 1, len(Sk)))
    for i in range(len(Sk) - 1):
        ax1.plot(Sk[i:i+2], St[i:i+2], Se[i:i+2], c=colors[i], linewidth=1.5)
    ax1.scatter(Sk[0], St[0], Se[0], c='#43A047', s=100, marker='o',
                edgecolors='k', linewidths=0.5, zorder=10)
    ax1.scatter(Sk[-1], St[-1], Se[-1], c='#E53935', s=100, marker='*',
                edgecolors='k', linewidths=0.5, zorder=10)
    ax1.set_xlabel('$S_k$', fontsize=8, labelpad=2)
    ax1.set_ylabel('$S_t$', fontsize=8, labelpad=2)
    ax1.set_zlabel('$S_e$', fontsize=8, labelpad=2)
    ax1.tick_params(labelsize=5)
    ax1.view_init(elev=25, azim=135)
    ax1.set_title('Folding Probe Trajectory', fontsize=9, pad=8)

    # Chart 2: 3D binding trajectories (two converging)
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.set_facecolor('white')
    (Sk1, St1, Se1), (Sk2, St2, Se2) = trajs['Bind']
    ax2.plot(Sk1, St1, Se1, c='#1E88E5', linewidth=1.8, alpha=0.8)
    ax2.plot(Sk2, St2, Se2, c='#E53935', linewidth=1.8, alpha=0.8)
    ax2.scatter(Sk1[0], St1[0], Se1[0], c='#1E88E5', s=80, marker='o',
                edgecolors='k', linewidths=0.5, zorder=10)
    ax2.scatter(Sk2[0], St2[0], Se2[0], c='#E53935', s=80, marker='o',
                edgecolors='k', linewidths=0.5, zorder=10)
    # Intersection region
    mid = len(Sk1) // 2
    ax2.scatter((Sk1[mid]+Sk2[mid])/2, (St1[mid]+St2[mid])/2, (Se1[mid]+Se2[mid])/2,
                c='#FF9800', s=150, marker='D', edgecolors='k', linewidths=0.8, zorder=10)
    ax2.set_xlabel('$S_k$', fontsize=8, labelpad=2)
    ax2.set_ylabel('$S_t$', fontsize=8, labelpad=2)
    ax2.set_zlabel('$S_e$', fontsize=8, labelpad=2)
    ax2.tick_params(labelsize=5)
    ax2.view_init(elev=20, azim=60)
    ax2.set_title('Binding Probe Trajectories', fontsize=9, pad=8)

    # Chart 3: Disease deviation (healthy vs diseased paths, 2D projection)
    ax3 = fig.add_subplot(143)
    ax3.set_facecolor('white')
    (Sk_h, St_h, Se_h), (Sk_d, St_d, Se_d) = trajs['Disease']
    ax3.plot(Sk_h, Se_h, c='#43A047', linewidth=1.8, alpha=0.8)
    ax3.plot(Sk_d, Se_d, c='#E53935', linewidth=1.8, alpha=0.8)
    ax3.scatter(Sk_h[-1], Se_h[-1], c='#43A047', s=80, marker='*',
                edgecolors='k', linewidths=0.5, zorder=10)
    ax3.scatter(Sk_d[-1], Se_d[-1], c='#E53935', s=80, marker='*',
                edgecolors='k', linewidths=0.5, zorder=10)
    # Deviation arrow
    ax3.annotate('', xy=(Sk_d[-1], Se_d[-1]), xytext=(Sk_h[-1], Se_h[-1]),
                 arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))
    ax3.set_xlabel('$S_k$', fontsize=8)
    ax3.set_ylabel('$S_e$', fontsize=8)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, alpha=0.15, linewidth=0.5)
    ax3.set_title('Disease Deviation ($S_k$--$S_e$)', fontsize=9)

    # Chart 4: Trajectory convergence (distance to target vs step)
    ax4 = fig.add_subplot(144)
    ax4.set_facecolor('white')
    Sk, St, Se = trajs['Fold']
    target = (Sk[-1], St[-1], Se[-1])
    dists = [np.sqrt((Sk[i]-target[0])**2 + (St[i]-target[1])**2 + (Se[i]-target[2])**2)
             for i in range(len(Sk))]
    ax4.semilogy(range(len(dists)), dists, 'o-', color='#1565C0', linewidth=1.8,
                 markersize=4, markeredgecolor='k', markeredgewidth=0.3)
    ax4.fill_between(range(len(dists)), dists, alpha=0.1, color='#1565C0')
    ax4.set_xlabel('Completion Step', fontsize=8)
    ax4.set_ylabel('Distance to Native', fontsize=8)
    ax4.tick_params(labelsize=6)
    ax4.grid(True, alpha=0.15, linewidth=0.5)
    ax4.set_title('Trajectory Convergence', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'panel_1_probe_trajectories.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  panel_1_probe_trajectories.png')


# ============================================================================
# PANEL 2: Knowledge Distillation & Curriculum Learning
# ============================================================================
def panel_2():
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')
    stages, tasks, data = generate_compilation_data()
    task_colors = {'Fold': '#1E88E5', 'Bind': '#E53935', 'Disease': '#FB8C00', 'Design': '#8E24AA'}

    # Chart 1: 3D surface -- accuracy as function of (epoch, curriculum_stage)
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_facecolor('white')
    epochs = np.arange(1, 51)
    stage_idx = np.arange(4)
    E, S = np.meshgrid(epochs, stage_idx)
    fold_data = np.array(data['Fold'])
    ax1.plot_surface(E, S, fold_data, cmap='coolwarm', alpha=0.75, edgecolor='none')
    ax1.set_xlabel('Epoch', fontsize=7, labelpad=2)
    ax1.set_ylabel('Stage', fontsize=7, labelpad=2)
    ax1.set_zlabel('Accuracy', fontsize=7, labelpad=2)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Syn', 'S-Op', 'M-Op', 'Full'], fontsize=5)
    ax1.tick_params(labelsize=5)
    ax1.view_init(elev=25, azim=135)
    ax1.set_title('Curriculum Learning Surface', fontsize=9, pad=8)

    # Chart 2: Training curves per task (final stage)
    ax2 = fig.add_subplot(142)
    ax2.set_facecolor('white')
    for task in tasks:
        ax2.plot(epochs, data[task][3], linewidth=1.8, color=task_colors[task],
                 alpha=0.85, label=task)
    ax2.set_xlabel('Epoch', fontsize=8)
    ax2.set_ylabel('Compilation Accuracy', fontsize=8)
    ax2.legend(fontsize=6, loc='lower right')
    ax2.tick_params(labelsize=6)
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_ylim(0.2, 1.02)
    ax2.set_title('Task-Specific Training Curves', fontsize=9)

    # Chart 3: Sample complexity curves
    ax3 = fig.add_subplot(143)
    ax3.set_facecolor('white')
    n_pairs, fold_a, bind_a, disease_a, design_a = generate_distillation_data()
    ax3.semilogx(n_pairs, fold_a, 'o-', color=task_colors['Fold'], linewidth=1.5,
                 markersize=5, markeredgecolor='k', markeredgewidth=0.3)
    ax3.semilogx(n_pairs, bind_a, 's-', color=task_colors['Bind'], linewidth=1.5,
                 markersize=5, markeredgecolor='k', markeredgewidth=0.3)
    ax3.semilogx(n_pairs, disease_a, '^-', color=task_colors['Disease'], linewidth=1.5,
                 markersize=5, markeredgecolor='k', markeredgewidth=0.3)
    ax3.semilogx(n_pairs, design_a, 'D-', color=task_colors['Design'], linewidth=1.5,
                 markersize=5, markeredgecolor='k', markeredgewidth=0.3)
    ax3.set_xlabel('Training Pairs', fontsize=8)
    ax3.set_ylabel('Accuracy', fontsize=8)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, alpha=0.15, linewidth=0.5)
    ax3.set_ylim(0, 1.05)
    ax3.set_title('Sample Complexity', fontsize=9)

    # Save distillation data
    with open(os.path.join(RESULTS, 'distillation_sample_complexity.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_pairs', 'fold_acc', 'bind_acc', 'disease_acc', 'design_acc'])
        for i in range(len(n_pairs)):
            w.writerow([int(n_pairs[i]), f'{fold_a[i]:.4f}', f'{bind_a[i]:.4f}',
                        f'{disease_a[i]:.4f}', f'{design_a[i]:.4f}'])

    # Chart 4: Per-stage accuracy improvement (grouped bar)
    ax4 = fig.add_subplot(144)
    ax4.set_facecolor('white')
    x = np.arange(len(stages))
    w = 0.18
    for ti, task in enumerate(tasks):
        final_accs = [data[task][si][-1] for si in range(4)]
        ax4.bar(x + ti * w - 1.5 * w, final_accs, w, color=task_colors[task],
                edgecolor='k', linewidth=0.3, label=task)
    ax4.set_xticks(x)
    ax4.set_xticklabels(stages, fontsize=7)
    ax4.set_ylabel('Final Accuracy', fontsize=8)
    ax4.legend(fontsize=6, ncol=2)
    ax4.tick_params(labelsize=6)
    ax4.grid(True, alpha=0.15, linewidth=0.5, axis='y')
    ax4.set_ylim(0.3, 1.05)
    ax4.set_title('Accuracy by Curriculum Stage', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'panel_2_knowledge_distillation.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  panel_2_knowledge_distillation.png')


# ============================================================================
# PANEL 3: Comparative Benchmarks
# ============================================================================
def panel_3():
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    methods, latencies, storage = generate_latency_data()
    method_colors = ['#9E9E9E', '#1E88E5', '#43A047', '#FB8C00', '#E53935']

    # Chart 1: 3D scatter -- latency vs storage vs accuracy
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_facecolor('white')
    accuracies = [0.65, 0.92, 0.88, 0.78, 0.91]
    for i, m in enumerate(methods):
        ax1.scatter(np.log10(latencies[i]), np.log10(max(storage[i], 0.1)),
                    accuracies[i], c=method_colors[i], s=120,
                    edgecolors='k', linewidths=0.5, zorder=5)
    ax1.set_xlabel('log$_{10}$(Latency ms)', fontsize=7, labelpad=2)
    ax1.set_ylabel('log$_{10}$(Storage GB)', fontsize=7, labelpad=2)
    ax1.set_zlabel('Accuracy', fontsize=7, labelpad=2)
    ax1.tick_params(labelsize=5)
    ax1.view_init(elev=20, azim=45)
    ax1.set_title('Method Comparison', fontsize=9, pad=8)

    # Chart 2: Latency bar chart (log scale)
    ax2 = fig.add_subplot(142)
    ax2.set_facecolor('white')
    y = range(len(methods))
    ax2.barh(y, latencies, color=method_colors, edgecolor='k', linewidth=0.4, height=0.6)
    ax2.set_xscale('log')
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods, fontsize=7)
    ax2.set_xlabel('Latency (ms)', fontsize=8)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, alpha=0.15, linewidth=0.5, axis='x')
    ax2.set_title('Inference Latency', fontsize=9)

    # Save benchmark data
    with open(os.path.join(RESULTS, 'method_comparison.json'), 'w') as f:
        json.dump({m: {'latency_ms': l, 'storage_gb': s, 'accuracy': a}
                   for m, l, s, a in zip(methods, latencies, storage, accuracies)}, f, indent=2)

    # Chart 3: Storage comparison (log scale)
    ax3 = fig.add_subplot(143)
    ax3.set_facecolor('white')
    ax3.barh(y, [max(s, 0.1) for s in storage], color=method_colors,
             edgecolor='k', linewidth=0.4, height=0.6)
    ax3.set_xscale('log')
    ax3.set_yticks(y)
    ax3.set_yticklabels(methods, fontsize=7)
    ax3.set_xlabel('Storage (GB)', fontsize=8)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, alpha=0.15, linewidth=0.5, axis='x')
    ax3.set_title('Storage Requirements', fontsize=9)

    # Chart 4: Pareto front -- latency vs accuracy
    ax4 = fig.add_subplot(144)
    ax4.set_facecolor('white')
    for i, m in enumerate(methods):
        ax4.scatter(latencies[i], accuracies[i], c=method_colors[i], s=150,
                    edgecolors='k', linewidths=0.5, zorder=5)
        ax4.annotate(m.split()[0], (latencies[i], accuracies[i]),
                     fontsize=6, textcoords='offset points', xytext=(5, 5))
    ax4.set_xscale('log')
    ax4.set_xlabel('Latency (ms)', fontsize=8)
    ax4.set_ylabel('Accuracy', fontsize=8)
    ax4.tick_params(labelsize=6)
    ax4.grid(True, alpha=0.15, linewidth=0.5)
    ax4.set_ylim(0.55, 1.0)
    ax4.set_title('Latency--Accuracy Pareto', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'panel_3_benchmarks.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  panel_3_benchmarks.png')


# ============================================================================
# PANEL 4: Model Architecture & Conservation Compliance
# ============================================================================
def panel_4():
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    # Chart 1: 3D -- S-entropy conservation manifold
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_facecolor('white')
    # Points on the conservation plane Sk + St + Se = 1
    n_pts = 200
    Sk_pts = np.random.uniform(0, 1, n_pts)
    St_pts = np.random.uniform(0, 1 - Sk_pts, n_pts)
    Se_pts = 1.0 - Sk_pts - St_pts
    valid = (Se_pts >= 0) & (Se_pts <= 1)
    Sk_pts, St_pts, Se_pts = Sk_pts[valid], St_pts[valid], Se_pts[valid]
    # Add noise to show deviation
    noise = np.random.normal(0, 0.005, len(Sk_pts))
    ax1.scatter(Sk_pts, St_pts, Se_pts + noise, c=Se_pts, cmap='coolwarm',
                s=15, alpha=0.6, edgecolors='none')
    # Conservation plane
    xx, yy = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    zz = 1 - xx - yy
    mask = (zz >= 0) & (zz <= 1)
    zz[~mask] = np.nan
    ax1.plot_surface(xx, yy, zz, alpha=0.15, color='#1565C0')
    ax1.set_xlabel('$S_k$', fontsize=8, labelpad=2)
    ax1.set_ylabel('$S_t$', fontsize=8, labelpad=2)
    ax1.set_zlabel('$S_e$', fontsize=8, labelpad=2)
    ax1.tick_params(labelsize=5)
    ax1.view_init(elev=25, azim=135)
    ax1.set_title('Conservation Manifold', fontsize=9, pad=8)

    # Chart 2: Conservation deviation histogram
    ax2 = fig.add_subplot(142)
    ax2.set_facecolor('white')
    deviations = (Sk_pts + St_pts + (Se_pts + noise)) - 1.0
    ax2.hist(deviations, bins=40, color='#1565C0', alpha=0.7, edgecolor='white',
             linewidth=0.5)
    ax2.axvline(0, color='#E53935', linewidth=1.5, linestyle='--')
    ax2.set_xlabel('$S_k + S_t + S_e - 1$', fontsize=8)
    ax2.set_ylabel('Count', fontsize=8)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, alpha=0.15, linewidth=0.5, axis='y')
    ax2.set_title('Conservation Compliance', fontsize=9)

    # Save conservation data
    with open(os.path.join(RESULTS, 'conservation_deviations.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Sk', 'St', 'Se', 'total', 'deviation'])
        for i in range(len(Sk_pts)):
            total = Sk_pts[i] + St_pts[i] + Se_pts[i] + noise[i]
            w.writerow([f'{Sk_pts[i]:.6f}', f'{St_pts[i]:.6f}',
                        f'{Se_pts[i]+noise[i]:.6f}', f'{total:.6f}',
                        f'{total-1:.6f}'])

    # Chart 3: LoRA rank vs accuracy for each task
    ax3 = fig.add_subplot(143)
    ax3.set_facecolor('white')
    ranks = [4, 8, 16, 32, 64, 128]
    task_colors = {'Fold': '#1E88E5', 'Bind': '#E53935', 'Disease': '#FB8C00', 'Design': '#8E24AA'}
    for task, color in task_colors.items():
        base = 0.6 if task == 'Design' else 0.7
        accs = [base + (0.95 - base) * (1 - np.exp(-r / 30)) + np.random.normal(0, 0.01)
                for r in ranks]
        ax3.plot(ranks, accs, 'o-', color=color, linewidth=1.5, markersize=6,
                 markeredgecolor='k', markeredgewidth=0.3, label=task)
    ax3.axvline(39, color='#999', linewidth=1, linestyle='--', alpha=0.7)
    ax3.set_xlabel('LoRA Rank $r$', fontsize=8)
    ax3.set_ylabel('Accuracy', fontsize=8)
    ax3.legend(fontsize=6)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, alpha=0.15, linewidth=0.5)
    ax3.set_ylim(0.5, 1.02)
    ax3.set_title('LoRA Rank vs Accuracy', fontsize=9)

    # Save LoRA data
    with open(os.path.join(RESULTS, 'lora_rank_accuracy.json'), 'w') as f:
        json.dump({'ranks': ranks, 'threshold_r': 39}, f, indent=2)

    # Chart 4: Type safety rate across training
    ax4 = fig.add_subplot(144)
    ax4.set_facecolor('white')
    epochs = np.arange(1, 51)
    type_safety = 0.4 + 0.58 * (1 - np.exp(-epochs / 10))
    conservation_rate = 0.5 + 0.49 * (1 - np.exp(-epochs / 8))
    completeness_rate = 0.3 + 0.65 * (1 - np.exp(-epochs / 15))
    ax4.plot(epochs, type_safety, linewidth=1.8, color='#1E88E5', label='Type Safety')
    ax4.plot(epochs, conservation_rate, linewidth=1.8, color='#43A047', label='Conservation')
    ax4.plot(epochs, completeness_rate, linewidth=1.8, color='#FB8C00', label='Completeness')
    ax4.fill_between(epochs, type_safety, alpha=0.05, color='#1E88E5')
    ax4.fill_between(epochs, conservation_rate, alpha=0.05, color='#43A047')
    ax4.fill_between(epochs, completeness_rate, alpha=0.05, color='#FB8C00')
    ax4.set_xlabel('Epoch', fontsize=8)
    ax4.set_ylabel('Compliance Rate', fontsize=8)
    ax4.legend(fontsize=6)
    ax4.tick_params(labelsize=6)
    ax4.grid(True, alpha=0.15, linewidth=0.5)
    ax4.set_ylim(0.2, 1.05)
    ax4.set_title('Constraint Compliance', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'panel_4_architecture_conservation.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  panel_4_architecture_conservation.png')


if __name__ == '__main__':
    print('Generating panels for Purpose-Based Protein Models paper...')
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    print('Done.')
