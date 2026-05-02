"""
Plot behavioral metric comparisons between trained agents and human data.
Generates five figures, one per metric, plus overall helping rate and fit score heatmap.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme
AGENT_COLOR = 'steelblue'
HUMAN_COLOR = 'lightcoral'
LINE_COLORS = {'Yes': 'orange', 'No': 'teal'}

REWARD_MODES = ['selfish', 'capacity', 'proximity', 'reciprocity', 'capacity_proximity']
FIGURE_DIR = 'figures'


def get_reward_mode_title(reward_mode):
    if reward_mode == 'capacity_proximity':
        return 'Cap+Prox'
    return reward_mode.capitalize()


def ensure_figures_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def load_metric_csv(metric_num, reward_mode, tag=""):
    suffix = f"_{tag}" if tag else ""
    names = {
        1: f"results/metrics_{reward_mode}{suffix}_metric1_backpack.csv",
        2: f"results/metrics_{reward_mode}{suffix}_metric2_patchuniformity.csv",
        3: f"results/metrics_{reward_mode}{suffix}_metric3_distance.csv",
        4: f"results/metrics_{reward_mode}{suffix}_metric4_energy.csv",
        5: f"results/metrics_{reward_mode}{suffix}_metric5_reciprocity.csv",
    }
    filename = names.get(metric_num)
    if filename and os.path.exists(filename):
        return pd.read_csv(filename)
    return None


def plot_metric_1(tag=""):
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    fig.suptitle('Helping rate by backpack size: Agent vs. Human', fontsize=14, fontweight='bold')
    all_rates, data_map = [], {}
    for m in REWARD_MODES:
        df = load_metric_csv(1, m, tag)
        if df is not None:
            data_map[m] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    for i, m in enumerate(REWARD_MODES):
        ax, df = axes[i], data_map.get(m)
        if df is not None:
            x, w = np.arange(len(df)), 0.35
            ax.bar(x - w/2, df['agent_helping_rate'], w, label='Agent', color=AGENT_COLOR)
            ax.bar(x + w/2, df['human_helping_rate'], w, label='Human', color=HUMAN_COLOR)
            ax.set_xlabel('Backpack Size'); ax.set_ylabel('Helping Rate')
            ax.set_title(get_reward_mode_title(m))
            ax.set_xticks(x); ax.set_xticklabels(df['backpack_size'].astype(int))
            ax.set_ylim(0, ylim_max)
            if i == 0: ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(get_reward_mode_title(m))
    ax_h = axes[5]
    hd = next((data_map[m] for m in REWARD_MODES if m in data_map), None)
    if hd is not None:
        x = np.arange(len(hd))
        ax_h.bar(x, hd['human_helping_rate'], width=0.35, color=HUMAN_COLOR)
        ax_h.set_xlabel('Backpack Size'); ax_h.set_ylabel('Helping Rate'); ax_h.set_title('Human')
        ax_h.set_xticks(x); ax_h.set_xticklabels(hd['backpack_size'].astype(int))
        ax_h.set_ylim(0, ylim_max)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_metric1_backpack.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); print(f"Saved {path}"); plt.close()


def plot_metric_2(tag=""):
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    fig.suptitle('Helping rate by patch uniformity: Agent vs. Human', fontsize=14, fontweight='bold')
    all_rates, data_map = [], {}
    for m in REWARD_MODES:
        df = load_metric_csv(2, m, tag)
        if df is not None:
            data_map[m] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    for i, m in enumerate(REWARD_MODES):
        ax, df = axes[i], data_map.get(m)
        if df is not None:
            x, w = np.arange(len(df)), 0.35
            ax.bar(x - w/2, df['agent_helping_rate'], w, label='Agent', color=AGENT_COLOR)
            ax.bar(x + w/2, df['human_helping_rate'], w, label='Human', color=HUMAN_COLOR)
            ax.set_xlabel('Patch Uniformity'); ax.set_ylabel('Helping Rate')
            ax.set_title(get_reward_mode_title(m))
            ax.set_xticks(x); ax.set_xticklabels(df['patchUniformity'], rotation=30, ha='right')
            ax.set_ylim(0, ylim_max)
            if i == 0: ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(get_reward_mode_title(m))
    ax_h = axes[5]
    hd = next((data_map[m] for m in REWARD_MODES if m in data_map), None)
    if hd is not None:
        x = np.arange(len(hd))
        ax_h.bar(x, hd['human_helping_rate'], width=0.35, color=HUMAN_COLOR)
        ax_h.set_xlabel('Patch Uniformity'); ax_h.set_ylabel('Helping Rate'); ax_h.set_title('Human')
        ax_h.set_xticks(x); ax_h.set_xticklabels(hd['patchUniformity'], rotation=30, ha='right')
        ax_h.set_ylim(0, ylim_max)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_metric2_patchuniformity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); print(f"Saved {path}"); plt.close()


def plot_metric_3(tag=""):
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    fig.suptitle('Helping rate by distance to partner vegetables: Agent vs. Human', fontsize=14, fontweight='bold')
    all_rates, data_map = [], {}
    for m in REWARD_MODES:
        df = load_metric_csv(3, m, tag)
        if df is not None:
            data_map[m] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    for i, m in enumerate(REWARD_MODES):
        ax, df = axes[i], data_map.get(m)
        if df is not None:
            x, w = np.arange(len(df)), 0.35
            ax.bar(x - w/2, df['agent_helping_rate'], w, label='Agent', color=AGENT_COLOR)
            ax.bar(x + w/2, df['human_helping_rate'], w, label='Human', color=HUMAN_COLOR)
            ax.set_xlabel('Distance Bin'); ax.set_ylabel('Helping Rate')
            ax.set_title(get_reward_mode_title(m))
            ax.set_xticks(x); ax.set_xticklabels(df['distance_bin'], rotation=30, ha='right')
            ax.set_ylim(0, ylim_max)
            if i == 0: ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(get_reward_mode_title(m))
    ax_h = axes[5]
    hd = next((data_map[m] for m in REWARD_MODES if m in data_map), None)
    if hd is not None:
        x = np.arange(len(hd))
        ax_h.bar(x, hd['human_helping_rate'], width=0.35, color=HUMAN_COLOR)
        ax_h.set_xlabel('Distance Bin'); ax_h.set_ylabel('Helping Rate'); ax_h.set_title('Human')
        ax_h.set_xticks(x); ax_h.set_xticklabels(hd['distance_bin'], rotation=30, ha='right')
        ax_h.set_ylim(0, ylim_max)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_metric3_distance.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); print(f"Saved {path}"); plt.close()


def plot_metric_4(tag=""):
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    fig.suptitle('Helping rate by remaining energy: Agent vs. Human', fontsize=14, fontweight='bold')
    all_rates, data_map = [], {}
    for m in REWARD_MODES:
        df = load_metric_csv(4, m, tag)
        if df is not None:
            data_map[m] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    for i, m in enumerate(REWARD_MODES):
        ax, df = axes[i], data_map.get(m)
        if df is not None:
            x, w = np.arange(len(df)), 0.35
            ax.bar(x - w/2, df['agent_helping_rate'], w, label='Agent', color=AGENT_COLOR)
            ax.bar(x + w/2, df['human_helping_rate'], w, label='Human', color=HUMAN_COLOR)
            ax.set_xlabel('Energy Bin'); ax.set_ylabel('Helping Rate')
            ax.set_title(get_reward_mode_title(m))
            ax.set_xticks(x); ax.set_xticklabels(df['energy_bin'], rotation=45, ha='right')
            ax.set_ylim(0, ylim_max)
            if i == 0: ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(get_reward_mode_title(m))
    ax_h = axes[5]
    hd = next((data_map[m] for m in REWARD_MODES if m in data_map), None)
    if hd is not None:
        x = np.arange(len(hd))
        ax_h.bar(x, hd['human_helping_rate'], width=0.35, color=HUMAN_COLOR)
        ax_h.set_xlabel('Energy Bin'); ax_h.set_ylabel('Helping Rate'); ax_h.set_title('Human')
        ax_h.set_xticks(x); ax_h.set_xticklabels(hd['energy_bin'], rotation=45, ha='right')
        ax_h.set_ylim(0, ylim_max)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_metric4_energy.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); print(f"Saved {path}"); plt.close()


def plot_metric_5(tag=""):
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    fig.suptitle('Helping rate by turn (conditional on partner help): Agent vs. Human', fontsize=14, fontweight='bold')
    all_rates, data_map = [], {}
    for m in REWARD_MODES:
        df = load_metric_csv(5, m, tag)
        if df is not None:
            data_map[m] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    for i, m in enumerate(REWARD_MODES):
        ax, df = axes[i], data_map.get(m)
        if df is not None:
            df_yes = df[df['partner_helped_last'] == 'Yes'].sort_values('turn')
            df_no  = df[df['partner_helped_last'] == 'No'].sort_values('turn')
            if len(df_yes) > 0:
                ax.plot(df_yes['turn'] + 1, df_yes['agent_helping_rate'],
                        color=LINE_COLORS['Yes'], linestyle='-', marker='o', label='Agent: Partner helped')
                ax.plot(df_yes['turn'] + 1, df_yes['human_helping_rate'],
                        color=LINE_COLORS['Yes'], linestyle='--', marker='o', alpha=0.6, label='Human: Partner helped')
            if len(df_no) > 0:
                ax.plot(df_no['turn'] + 1, df_no['agent_helping_rate'],
                        color=LINE_COLORS['No'], linestyle='-', marker='s', label='Agent: Partner did not help')
                ax.plot(df_no['turn'] + 1, df_no['human_helping_rate'],
                        color=LINE_COLORS['No'], linestyle='--', marker='s', alpha=0.6, label='Human: Partner did not help')
            ax.set_xlabel('Turn'); ax.set_ylabel('Helping Rate')
            ax.set_title(get_reward_mode_title(m))
            ax.set_xticks(range(1, 11)); ax.set_ylim(0, ylim_max)
            if i == 0: ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(get_reward_mode_title(m))
    ax_h = axes[5]
    hd = next((data_map[m] for m in REWARD_MODES if m in data_map), None)
    if hd is not None:
        df_yes = hd[hd['partner_helped_last'] == 'Yes'].sort_values('turn')
        df_no  = hd[hd['partner_helped_last'] == 'No'].sort_values('turn')
        if len(df_yes) > 0:
            ax_h.plot(df_yes['turn'] + 1, df_yes['human_helping_rate'],
                      color=LINE_COLORS['Yes'], linestyle='--', marker='o', label='Partner helped')
        if len(df_no) > 0:
            ax_h.plot(df_no['turn'] + 1, df_no['human_helping_rate'],
                      color=LINE_COLORS['No'], linestyle='--', marker='s', label='Partner did not help')
        ax_h.set_xlabel('Turn'); ax_h.set_ylabel('Helping Rate'); ax_h.set_title('Human')
        ax_h.set_xticks(range(1, 11)); ax_h.set_ylim(0, ylim_max)
        ax_h.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_metric5_reciprocity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); print(f"Saved {path}"); plt.close()


def plot_overall_helping_rate(tag=""):
    human_rate = None
    agent_rates = {}
    for m in REWARD_MODES:
        df = load_metric_csv(1, m, tag)
        if df is not None:
            agent_rates[m] = df['agent_helping_rate'].mean()
            if human_rate is None:
                human_rate = df['human_helping_rate'].mean()
    if not agent_rates:
        print("No data for overall helping rate plot.")
        return
    labels = [get_reward_mode_title(m) for m in REWARD_MODES if m in agent_rates]
    values = [agent_rates[m] for m in REWARD_MODES if m in agent_rates]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=AGENT_COLOR, label='Agent')
    if human_rate is not None:
        ax.axhline(human_rate, color='lightcoral', linestyle='--', linewidth=2,
                   label=f'Human ({human_rate:.3f})')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Overall Helping Rate')
    ax.set_title('Overall Helping Rate: Agent vs. Human Baseline')
    ax.legend()
    ax.set_ylim(0, max(values + ([human_rate] if human_rate else [])) * 1.2)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_overall_helping_rate.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); print(f"Saved {path}"); plt.close()


def plot_fit_scores(tag=""):
    from scipy.stats import pearsonr
    metric_labels = ['Backpack', 'Patch\nUniformity', 'Distance', 'Energy', 'Reciprocity']
    scores = {}
    for m in REWARD_MODES:
        row = []
        for n in range(1, 6):
            df = load_metric_csv(n, m, tag)
            if df is None or len(df) < 2:
                row.append(np.nan); continue
            if n == 5:
                df = df.groupby('partner_helped_last')[['agent_helping_rate', 'human_helping_rate']].mean().reset_index()
            av, hv = df['agent_helping_rate'].values, df['human_helping_rate'].values
            if len(av) < 2 or np.std(av) == 0 or np.std(hv) == 0:
                row.append(np.nan)
            else:
                r, _ = pearsonr(av, hv)
                row.append(round(r, 3))
        scores[m] = row
    mode_labels = [get_reward_mode_title(m) for m in REWARD_MODES]
    matrix = np.array([scores[m] for m in REWARD_MODES], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(metric_labels))); ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(mode_labels))); ax.set_yticklabels(mode_labels, fontsize=10)
    ax.set_title("Fit Score (Pearson r): Agent vs. Human per Metric", fontsize=12, fontweight='bold')
    for i in range(len(REWARD_MODES)):
        for j in range(len(metric_labels)):
            val = matrix[i, j]
            text = f'{val:.2f}' if not np.isnan(val) else 'N/A'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='black' if np.isnan(val) or abs(val) < 0.7 else 'white')
    plt.colorbar(im, ax=ax, label='Pearson r')
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_fit_scores.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); print(f"Saved {path}"); plt.close()
    print("\nFit Score Summary (Pearson r):")
    header = f"{'Mode':<20}" + "".join(f"{l.replace(chr(10),' '):>16}" for l in metric_labels) + f"{'Mean r':>10}"
    print(header); print("-" * len(header))
    for mode, row in zip(mode_labels, matrix):
        mean_r = np.nanmean(row)
        print(f"{mode:<20}" + "".join(f"{v:>16.3f}" if not np.isnan(v) else f"{'N/A':>16}" for v in row) + f"{mean_r:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Plot behavioral metric figures")
    parser.add_argument("--tag", type=str, default="", help="Results tag, e.g. 'v4'")
    args = parser.parse_args()
    tag = args.tag
    ensure_figures_dir()
    print(f"Generating figures (tag='{tag}')...")
    plot_metric_1(tag=tag)
    plot_metric_2(tag=tag)
    plot_metric_3(tag=tag)
    plot_metric_4(tag=tag)
    plot_metric_5(tag=tag)
    plot_overall_helping_rate(tag=tag)
    plot_fit_scores(tag=tag)
    print("Done!")


if __name__ == '__main__':
    main()
