"""
Generate 4 publication panels for the Categorical Protein Database paper.
Each panel: 4 charts in a row, white background, minimal text, at least one 3D chart.
All data-driven -- no conceptual/text/table charts.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'validation', 'results')
FIGURES = os.path.join(os.path.dirname(__file__), '..', 'figures')

# ============================================================================
# Load all validation data
# ============================================================================
df_aa = pd.read_csv(os.path.join(RESULTS, 'amino_acid_coordinates.csv'))
df_pairs = pd.read_csv(os.path.join(RESULTS, 'pairwise_distances.csv'))
df_res = pd.read_csv(os.path.join(RESULTS, 'resolution_by_depth.csv'))
df_ss = pd.read_csv(os.path.join(RESULTS, 'secondary_structure_coordinates.csv'))
df_prot = pd.read_csv(os.path.join(RESULTS, 'protein_sentropy_coordinates.csv'))
df_scale = pd.read_csv(os.path.join(RESULTS, 'trajectory_completion_scaling.csv'))
with open(os.path.join(RESULTS, 'chemical_family_clustering.json')) as f:
    clustering = json.load(f)
with open(os.path.join(RESULTS, 'fold_class_separation.json')) as f:
    fold_sep = json.load(f)

CLASS_COLORS = {
    "nonpolar": "#3F51B5", "polar": "#4CAF50", "positive": "#F44336",
    "negative": "#FF9800", "aromatic": "#9C27B0",
}
FOLD_COLORS = {
    "all-alpha": "#E53935", "all-beta": "#1E88E5", "alpha+beta": "#8E24AA",
    "alpha/beta": "#FB8C00", "small-beta": "#43A047",
}

# ============================================================================
# PANEL 1: Amino Acid S-Entropy Space
# ============================================================================
def panel_1():
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5),
                              subplot_kw={}, gridspec_kw={'wspace': 0.28})
    fig.patch.set_facecolor('white')

    # Chart 1: 3D scatter of amino acids in S-entropy space
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_facecolor('white')
    for _, row in df_aa.iterrows():
        c = CLASS_COLORS.get(row['class'], '#999')
        ax1.scatter(row['Sk'], row['St'], row['Se'], c=c, s=90,
                    edgecolors='k', linewidths=0.4, alpha=0.9, zorder=5)
        ax1.text(row['Sk'], row['St'], row['Se'], f' {row["code1"]}',
                 fontsize=5.5, alpha=0.8)
    ax1.set_xlabel('$S_k$', fontsize=8, labelpad=2)
    ax1.set_ylabel('$S_t$', fontsize=8, labelpad=2)
    ax1.set_zlabel('$S_e$', fontsize=8, labelpad=2)
    ax1.set_xlim(0, 1.05); ax1.set_ylim(0, 1.05); ax1.set_zlim(0, 1.05)
    ax1.tick_params(labelsize=6)
    ax1.view_init(elev=25, azim=135)
    ax1.set_title('Amino Acids in $\\mathcal{S}$-Space', fontsize=9, pad=8)

    # Chart 2: Sk vs St projection with Voronoi-like scatter
    ax2 = axes[1]
    ax2.set_facecolor('white')
    for _, row in df_aa.iterrows():
        c = CLASS_COLORS.get(row['class'], '#999')
        ax2.scatter(row['Sk'], row['St'], c=c, s=100,
                    edgecolors='k', linewidths=0.4, zorder=5)
        ax2.annotate(row['code1'], (row['Sk'], row['St']),
                     fontsize=6, ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')
    ax2.set_xlabel('$S_k$ (hydrophobicity)', fontsize=8)
    ax2.set_ylabel('$S_t$ (volume)', fontsize=8)
    ax2.set_xlim(-0.05, 1.1); ax2.set_ylim(-0.05, 1.1)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_title('$S_k$--$S_t$ Projection', fontsize=9)

    # Chart 3: Resolution curve (pairs resolved vs depth)
    ax3 = axes[2]
    ax3.set_facecolor('white')
    ax3.fill_between(df_res['depth'], df_res['resolution_pct'],
                     alpha=0.15, color='#1565C0')
    ax3.plot(df_res['depth'], df_res['resolution_pct'], 'o-',
             color='#1565C0', linewidth=2, markersize=7, markeredgecolor='k',
             markeredgewidth=0.5)
    ax3.axhline(100, color='#999', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('Trit Depth $k$', fontsize=8)
    ax3.set_ylabel('Pairs Resolved (%)', fontsize=8)
    ax3.set_ylim(92, 101.5)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, alpha=0.15, linewidth=0.5)
    ax3.set_title('Resolution vs Trit Depth', fontsize=9)

    # Chart 4: Pairwise distance histogram
    ax4 = axes[3]
    ax4.set_facecolor('white')
    ax4.hist(df_pairs['distance'], bins=30, color='#1565C0', alpha=0.7,
             edgecolor='white', linewidth=0.5)
    ax4.axvline(df_pairs['distance'].min(), color='#F44336', linewidth=1.5,
                linestyle='--', alpha=0.8)
    ax4.set_xlabel('Euclidean Distance in $\\mathcal{S}$', fontsize=8)
    ax4.set_ylabel('Pair Count', fontsize=8)
    ax4.tick_params(labelsize=6)
    ax4.grid(True, alpha=0.15, linewidth=0.5, axis='y')
    ax4.set_title('Pairwise Distance Distribution', fontsize=9)

    plt.savefig(os.path.join(FIGURES, 'panel_1_amino_acid_sentropy.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  panel_1_amino_acid_sentropy.png')


# ============================================================================
# PANEL 2: Chemical Family Clustering & Similarity
# ============================================================================
def panel_2():
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    # Chart 1: 3D scatter colored by Se (electrostatic axis)
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_facecolor('white')
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.coolwarm
    for _, row in df_aa.iterrows():
        ax1.scatter(row['Sk'], row['St'], row['Se'],
                    c=[cmap(norm(row['Se']))], s=90,
                    edgecolors='k', linewidths=0.4)
    ax1.set_xlabel('$S_k$', fontsize=8, labelpad=2)
    ax1.set_ylabel('$S_t$', fontsize=8, labelpad=2)
    ax1.set_zlabel('$S_e$', fontsize=8, labelpad=2)
    ax1.view_init(elev=20, azim=60)
    ax1.tick_params(labelsize=6)
    ax1.set_title('$S_e$ Gradient in $\\mathcal{S}$-Space', fontsize=9, pad=8)

    # Chart 2: Pairwise similarity heatmap (sorted by ternary address)
    ax2 = fig.add_subplot(142)
    ax2.set_facecolor('white')
    df_aa['ternary_18'] = df_aa['ternary_18'].astype(str).str.zfill(18)
    sorted_df = df_aa.sort_values('ternary_18')
    sorted_names = sorted_df['code1'].tolist()
    sorted_coords = list(zip(sorted_df['Sk'], sorted_df['St'], sorted_df['Se']))
    n = len(sorted_names)
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_mat[i, j] = 18
            else:
                t1 = str(sorted_df.iloc[i]['ternary_18'])
                t2 = str(sorted_df.iloc[j]['ternary_18'])
                spd = 0
                for k in range(min(len(t1), len(t2))):
                    if t1[k] != t2[k]:
                        break
                    spd = k + 1
                sim_mat[i, j] = spd
    im = ax2.imshow(sim_mat, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax2.set_xticks(range(n)); ax2.set_xticklabels(sorted_names, fontsize=5.5)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(sorted_names, fontsize=5.5)
    plt.colorbar(im, ax=ax2, shrink=0.75, pad=0.02)
    ax2.set_title('Ternary Similarity Matrix', fontsize=9)

    # Chart 3: Dendrogram from S-entropy distances
    ax3 = fig.add_subplot(143)
    ax3.set_facecolor('white')
    coords_arr = df_aa[['Sk', 'St', 'Se']].values
    dist_vec = pdist(coords_arr, metric='euclidean')
    Z = linkage(dist_vec, method='ward')
    class_colors_list = [CLASS_COLORS.get(c, '#999') for c in df_aa['class']]
    dn = dendrogram(Z, labels=df_aa['code1'].values, ax=ax3,
                    leaf_font_size=6, leaf_rotation=90,
                    above_threshold_color='#999')
    ax3.set_ylabel('Ward Distance', fontsize=8)
    ax3.tick_params(labelsize=6)
    ax3.set_title('Hierarchical Clustering', fontsize=9)

    # Chart 4: Cohesion ratio bar chart
    ax4 = fig.add_subplot(144)
    ax4.set_facecolor('white')
    cls_names = sorted(clustering.keys())
    Rs = [clustering[c]['cohesion_ratio_R'] for c in cls_names]
    bar_colors = [CLASS_COLORS.get(c, '#999') for c in cls_names]
    bars = ax4.barh(range(len(cls_names)), Rs, color=bar_colors, edgecolor='k',
                    linewidth=0.4, height=0.6)
    ax4.axvline(1.0, color='#999', linewidth=1, linestyle='--', alpha=0.7)
    ax4.set_yticks(range(len(cls_names)))
    ax4.set_yticklabels(cls_names, fontsize=7)
    ax4.set_xlabel('Cohesion Ratio $R$', fontsize=8)
    ax4.tick_params(labelsize=6)
    ax4.grid(True, alpha=0.15, linewidth=0.5, axis='x')
    ax4.set_title('Family Cohesion ($R > 1$ = cohesive)', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'panel_2_clustering_similarity.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  panel_2_clustering_similarity.png')


# ============================================================================
# PANEL 3: Secondary Structure Encoding
# ============================================================================
def panel_3():
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    SS_COLORS = {
        'alpha_helix': '#E53935', 'beta_sheet_parallel': '#1E88E5',
        'beta_sheet_antiparallel': '#0D47A1', '310_helix': '#FF7043',
        'random_coil': '#9E9E9E', 'beta_turn_type_I': '#43A047',
        'beta_turn_type_II': '#66BB6A', 'polyproline_II': '#FB8C00',
    }

    # Chart 1: 3D scatter of secondary structures
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_facecolor('white')
    for _, row in df_ss.iterrows():
        c = SS_COLORS.get(row['structure'], '#999')
        ax1.scatter(row['Sk'], row['St'], row['Se'], c=c, s=150,
                    edgecolors='k', linewidths=0.5, zorder=5)
    ax1.set_xlabel('$S_k$', fontsize=8, labelpad=2)
    ax1.set_ylabel('$S_t$', fontsize=8, labelpad=2)
    ax1.set_zlabel('$S_e$', fontsize=8, labelpad=2)
    ax1.view_init(elev=15, azim=45)
    ax1.tick_params(labelsize=6)
    ax1.set_title('Secondary Structures in $\\mathcal{S}$', fontsize=9, pad=8)

    # Chart 2: Amide I vs Amide III frequencies colored by structure type
    ax2 = fig.add_subplot(142)
    ax2.set_facecolor('white')
    for _, row in df_ss.iterrows():
        c = SS_COLORS.get(row['structure'], '#999')
        ax2.scatter(row['amide_I'], row['amide_III'], c=c, s=120,
                    edgecolors='k', linewidths=0.5, zorder=5)
    ax2.set_xlabel('Amide I (cm$^{-1}$)', fontsize=8)
    ax2.set_ylabel('Amide III (cm$^{-1}$)', fontsize=8)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_title('Amide I vs Amide III', fontsize=9)

    # Chart 3: Pairwise distance matrix for secondary structures
    ax3 = fig.add_subplot(143)
    ax3.set_facecolor('white')
    n = len(df_ss)
    coords = list(zip(df_ss['Sk'], df_ss['St'], df_ss['Se']))
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = np.sqrt(sum((a - b)**2 for a, b in zip(coords[i], coords[j])))
    im = ax3.imshow(dist_mat, cmap='viridis_r', aspect='auto', interpolation='nearest')
    short = [s[:8] for s in df_ss['structure']]
    ax3.set_xticks(range(n)); ax3.set_xticklabels(short, fontsize=5, rotation=45, ha='right')
    ax3.set_yticks(range(n)); ax3.set_yticklabels(short, fontsize=5)
    plt.colorbar(im, ax=ax3, shrink=0.75, pad=0.02)
    ax3.set_title('Pairwise Distance Matrix', fontsize=9)

    # Chart 4: St coordinate bar chart (the main discriminator)
    ax4 = fig.add_subplot(144)
    ax4.set_facecolor('white')
    sorted_ss = df_ss.sort_values('St')
    bar_colors = [SS_COLORS.get(s, '#999') for s in sorted_ss['structure']]
    ax4.barh(range(len(sorted_ss)), sorted_ss['St'], color=bar_colors,
             edgecolor='k', linewidth=0.4, height=0.6)
    ax4.set_yticks(range(len(sorted_ss)))
    ax4.set_yticklabels([s[:12] for s in sorted_ss['structure']], fontsize=6)
    ax4.set_xlabel('$S_t$ (timescale span)', fontsize=8)
    ax4.tick_params(labelsize=6)
    ax4.grid(True, alpha=0.15, linewidth=0.5, axis='x')
    ax4.set_title('$S_t$ Separation by Structure', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'panel_3_secondary_structure.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  panel_3_secondary_structure.png')


# ============================================================================
# PANEL 4: Protein Trajectories & Folding Complexity
# ============================================================================
def panel_4():
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')

    # Chart 1: 3D scatter of proteins colored by fold class
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_facecolor('white')
    for _, row in df_prot.iterrows():
        c = FOLD_COLORS.get(row['fold_class'], '#999')
        size = max(40, row['n_residues'] * 0.8)
        ax1.scatter(row['Sk'], row['St'], row['Se'], c=c, s=size,
                    edgecolors='k', linewidths=0.5, alpha=0.85, zorder=5)
    ax1.set_xlabel('$S_k$', fontsize=8, labelpad=2)
    ax1.set_ylabel('$S_t$', fontsize=8, labelpad=2)
    ax1.set_zlabel('$S_e$', fontsize=8, labelpad=2)
    ax1.view_init(elev=25, azim=120)
    ax1.tick_params(labelsize=6)
    ax1.set_title('Proteins in $\\mathcal{S}$-Space', fontsize=9, pad=8)

    # Chart 2: Trajectory complexity scaling (log-log)
    ax2 = fig.add_subplot(142)
    ax2.set_facecolor('white')
    Ns = df_scale['n_residues'].values
    steps = df_scale['trajectory_steps'].values
    ax2.loglog(Ns, steps, 'o-', color='#1565C0', linewidth=2, markersize=7,
               markeredgecolor='k', markeredgewidth=0.5, label='Trajectory $O(\\log_3 N)$')
    ax2.loglog(Ns, Ns, '--', color='#E53935', linewidth=1, alpha=0.5, label='$O(N)$ reference')
    ax2.loglog(Ns, np.sqrt(Ns), ':', color='#FB8C00', linewidth=1, alpha=0.5, label='$O(\\sqrt{N})$ reference')
    ax2.fill_between(Ns, steps, Ns, alpha=0.06, color='#1565C0')
    ax2.set_xlabel('Protein Length $N$', fontsize=8)
    ax2.set_ylabel('Steps', fontsize=8)
    ax2.legend(fontsize=6, loc='upper left')
    ax2.tick_params(labelsize=6)
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_title('Folding Complexity Scaling', fontsize=9)

    # Chart 3: Helix% vs Sheet% scatter colored by Sk
    ax3 = fig.add_subplot(143)
    ax3.set_facecolor('white')
    sc = ax3.scatter(df_prot['helix_pct'] * 100, df_prot['sheet_pct'] * 100,
                     c=df_prot['Sk'], cmap='RdYlBu', s=df_prot['n_residues'] * 0.6,
                     edgecolors='k', linewidths=0.5, zorder=5)
    plt.colorbar(sc, ax=ax3, shrink=0.75, pad=0.02, label='$S_k$')
    ax3.set_xlabel('Helix Content (%)', fontsize=8)
    ax3.set_ylabel('Sheet Content (%)', fontsize=8)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, alpha=0.15, linewidth=0.5)
    ax3.set_title('Structure Content vs $S_k$', fontsize=9)

    # Chart 4: Fold class centroid distances (heatmap)
    ax4 = fig.add_subplot(144)
    ax4.set_facecolor('white')
    fold_classes = sorted([k for k in fold_sep if '_vs_' not in k])
    n_fc = len(fold_classes)
    fc_dist = np.zeros((n_fc, n_fc))
    for i, c1 in enumerate(fold_classes):
        for j, c2 in enumerate(fold_classes):
            if i == j:
                fc_dist[i, j] = 0
            else:
                key1 = f"{c1}_vs_{c2}"
                key2 = f"{c2}_vs_{c1}"
                if key1 in fold_sep:
                    fc_dist[i, j] = fold_sep[key1]
                elif key2 in fold_sep:
                    fc_dist[i, j] = fold_sep[key2]
    im = ax4.imshow(fc_dist, cmap='Blues', aspect='auto', interpolation='nearest')
    ax4.set_xticks(range(n_fc))
    ax4.set_xticklabels([c[:8] for c in fold_classes], fontsize=6, rotation=45, ha='right')
    ax4.set_yticks(range(n_fc))
    ax4.set_yticklabels([c[:8] for c in fold_classes], fontsize=6)
    # Annotate cells
    for i in range(n_fc):
        for j in range(n_fc):
            ax4.text(j, i, f'{fc_dist[i,j]:.3f}', ha='center', va='center', fontsize=6,
                     color='white' if fc_dist[i, j] > 0.06 else 'black')
    plt.colorbar(im, ax=ax4, shrink=0.75, pad=0.02)
    ax4.set_title('Fold Class Centroid Distances', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'panel_4_protein_trajectories.png'),
                dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print('  panel_4_protein_trajectories.png')


if __name__ == '__main__':
    print('Generating panels for Categorical Protein Database paper...')
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    print('Done.')
