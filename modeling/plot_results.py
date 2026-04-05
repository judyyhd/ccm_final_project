"""
Plot behavioral metric comparisons between trained agents and human data.
Generates five figures, one per metric.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme
AGENT_COLOR = 'steelblue'
HUMAN_COLOR = 'lightcoral'
LINE_COLORS = {'Yes': 'orange', 'No': 'teal'}

REWARD_MODES = ['selfish', 'capacity', 'proximity', 'reciprocity']
FIGURE_DIR = 'figures'


def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    os.makedirs(FIGURE_DIR, exist_ok=True)


def load_metric_csv(metric_num, reward_mode):
    """Load a metric CSV for a given reward mode. Returns None if file doesn't exist."""
    if metric_num == 1:
        filename = f"results/metrics_{reward_mode}_metric1_backpack.csv"
    elif metric_num == 2:
        filename = f"results/metrics_{reward_mode}_metric2_patchuniformity.csv"
    elif metric_num == 3:
        filename = f"results/metrics_{reward_mode}_metric3_distance.csv"
    elif metric_num == 4:
        filename = f"results/metrics_{reward_mode}_metric4_energy.csv"
    elif metric_num == 5:
        filename = f"results/metrics_{reward_mode}_metric5_reciprocity.csv"
    else:
        return None
    
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return None


def plot_metric_1():
    """Plot Metric 1: Helping rate by backpack size."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Helping rate by backpack size: Agent vs. Human', fontsize=14, fontweight='bold')
    
    # Collect all data to determine y-axis limits
    all_rates = []
    data_map = {}
    for i, reward_mode in enumerate(REWARD_MODES):
        df = load_metric_csv(1, reward_mode)
        if df is not None:
            data_map[reward_mode] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    
    # Plot each reward mode
    for i, reward_mode in enumerate(REWARD_MODES):
        ax = axes[i]
        df = data_map.get(reward_mode)
        
        if df is not None:
            x = np.arange(len(df))
            width = 0.35
            
            ax.bar(x - width/2, df['agent_helping_rate'], width, label='Agent', color=AGENT_COLOR)
            ax.bar(x + width/2, df['human_helping_rate'], width, label='Human', color=HUMAN_COLOR)
            
            ax.set_xlabel('Backpack Size')
            ax.set_ylabel('Helping Rate')
            ax.set_title(reward_mode.capitalize())
            ax.set_xticks(x)
            ax.set_xticklabels(df['backpack_size'].astype(int))
            ax.set_ylim(0, ylim_max)
            
            if i == 0:
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(reward_mode.capitalize())
    
    # Human reference panel
    ax_human = axes[4]
    human_data = None
    for reward_mode in REWARD_MODES:
        df = data_map.get(reward_mode)
        if df is not None:
            human_data = df
            break
    
    if human_data is not None:
        x = np.arange(len(human_data))
        width = 0.35
        ax_human.bar(x, human_data['human_helping_rate'], width=width, label='Human', color=HUMAN_COLOR)
        ax_human.set_xlabel('Backpack Size')
        ax_human.set_ylabel('Helping Rate')
        ax_human.set_title('Human')
        ax_human.set_xticks(x)
        ax_human.set_xticklabels(human_data['backpack_size'].astype(int))
        ax_human.set_ylim(0, ylim_max)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'fig_metric1_backpack.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_metric_2():
    """Plot Metric 2: Helping rate by patch uniformity."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Helping rate by patch uniformity: Agent vs. Human', fontsize=14, fontweight='bold')
    
    # Collect all data
    all_rates = []
    data_map = {}
    for i, reward_mode in enumerate(REWARD_MODES):
        df = load_metric_csv(2, reward_mode)
        if df is not None:
            data_map[reward_mode] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    
    # Plot each reward mode
    for i, reward_mode in enumerate(REWARD_MODES):
        ax = axes[i]
        df = data_map.get(reward_mode)
        
        if df is not None:
            x = np.arange(len(df))
            width = 0.35
            
            ax.bar(x - width/2, df['agent_helping_rate'], width, label='Agent', color=AGENT_COLOR)
            ax.bar(x + width/2, df['human_helping_rate'], width, label='Human', color=HUMAN_COLOR)
            
            ax.set_xlabel('Patch Uniformity')
            ax.set_ylabel('Helping Rate')
            ax.set_title(reward_mode.capitalize())
            ax.set_xticks(x)
            ax.set_xticklabels(df['patchUniformity'], rotation=30, ha='right')
            ax.set_ylim(0, ylim_max)
            
            if i == 0:
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(reward_mode.capitalize())
    
    # Human reference panel
    ax_human = axes[4]
    human_data = None
    for reward_mode in REWARD_MODES:
        df = data_map.get(reward_mode)
        if df is not None:
            human_data = df
            break
    
    if human_data is not None:
        x = np.arange(len(human_data))
        width = 0.35
        ax_human.bar(x, human_data['human_helping_rate'], width=width, label='Human', color=HUMAN_COLOR)
        ax_human.set_xlabel('Patch Uniformity')
        ax_human.set_ylabel('Helping Rate')
        ax_human.set_title('Human')
        ax_human.set_xticks(x)
        ax_human.set_xticklabels(human_data['patchUniformity'], rotation=30, ha='right')
        ax_human.set_ylim(0, ylim_max)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'fig_metric2_patchuniformity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_metric_3():
    """Plot Metric 3: Helping rate by distance to partner vegetables."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Helping rate by distance to partner vegetables: Agent vs. Human', fontsize=14, fontweight='bold')
    
    # Collect all data
    all_rates = []
    data_map = {}
    for i, reward_mode in enumerate(REWARD_MODES):
        df = load_metric_csv(3, reward_mode)
        if df is not None:
            data_map[reward_mode] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    
    # Plot each reward mode
    for i, reward_mode in enumerate(REWARD_MODES):
        ax = axes[i]
        df = data_map.get(reward_mode)
        
        if df is not None:
            x = np.arange(len(df))
            width = 0.35
            
            ax.bar(x - width/2, df['agent_helping_rate'], width, label='Agent', color=AGENT_COLOR)
            ax.bar(x + width/2, df['human_helping_rate'], width, label='Human', color=HUMAN_COLOR)
            
            ax.set_xlabel('Distance Bin')
            ax.set_ylabel('Helping Rate')
            ax.set_title(reward_mode.capitalize())
            ax.set_xticks(x)
            ax.set_xticklabels(df['distance_bin'], rotation=30, ha='right')
            ax.set_ylim(0, ylim_max)
            
            if i == 0:
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(reward_mode.capitalize())
    
    # Human reference panel
    ax_human = axes[4]
    human_data = None
    for reward_mode in REWARD_MODES:
        df = data_map.get(reward_mode)
        if df is not None:
            human_data = df
            break
    
    if human_data is not None:
        x = np.arange(len(human_data))
        width = 0.35
        ax_human.bar(x, human_data['human_helping_rate'], width=width, label='Human', color=HUMAN_COLOR)
        ax_human.set_xlabel('Distance Bin')
        ax_human.set_ylabel('Helping Rate')
        ax_human.set_title('Human')
        ax_human.set_xticks(x)
        ax_human.set_xticklabels(human_data['distance_bin'], rotation=30, ha='right')
        ax_human.set_ylim(0, ylim_max)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'fig_metric3_distance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_metric_4():
    """Plot Metric 4: Helping rate by remaining energy."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Helping rate by remaining energy: Agent vs. Human', fontsize=14, fontweight='bold')
    
    # Collect all data
    all_rates = []
    data_map = {}
    for i, reward_mode in enumerate(REWARD_MODES):
        df = load_metric_csv(4, reward_mode)
        if df is not None:
            data_map[reward_mode] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    
    # Plot each reward mode
    for i, reward_mode in enumerate(REWARD_MODES):
        ax = axes[i]
        df = data_map.get(reward_mode)
        
        if df is not None:
            x = np.arange(len(df))
            width = 0.35
            
            ax.bar(x - width/2, df['agent_helping_rate'], width, label='Agent', color=AGENT_COLOR)
            ax.bar(x + width/2, df['human_helping_rate'], width, label='Human', color=HUMAN_COLOR)
            
            ax.set_xlabel('Energy Bin')
            ax.set_ylabel('Helping Rate')
            ax.set_title(reward_mode.capitalize())
            ax.set_xticks(x)
            ax.set_xticklabels(df['energy_bin'], rotation=45, ha='right')
            ax.set_ylim(0, ylim_max)
            
            if i == 0:
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(reward_mode.capitalize())
    
    # Human reference panel
    ax_human = axes[4]
    human_data = None
    for reward_mode in REWARD_MODES:
        df = data_map.get(reward_mode)
        if df is not None:
            human_data = df
            break
    
    if human_data is not None:
        x = np.arange(len(human_data))
        width = 0.35
        ax_human.bar(x, human_data['human_helping_rate'], width=width, label='Human', color=HUMAN_COLOR)
        ax_human.set_xlabel('Energy Bin')
        ax_human.set_ylabel('Helping Rate')
        ax_human.set_title('Human')
        ax_human.set_xticks(x)
        ax_human.set_xticklabels(human_data['energy_bin'], rotation=45, ha='right')
        ax_human.set_ylim(0, ylim_max)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'fig_metric4_energy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_metric_5():
    """Plot Metric 5: Helping rate by turn, conditional on partner help (line plot)."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Helping rate by turn (conditional on partner help): Agent vs. Human', fontsize=14, fontweight='bold')
    
    # Collect all data
    all_rates = []
    data_map = {}
    for i, reward_mode in enumerate(REWARD_MODES):
        df = load_metric_csv(5, reward_mode)
        if df is not None:
            data_map[reward_mode] = df
            all_rates.extend(df['agent_helping_rate'])
            all_rates.extend(df['human_helping_rate'])
    
    ylim_max = max(all_rates) * 1.15 if all_rates else 0.5
    
    # Plot each reward mode
    for i, reward_mode in enumerate(REWARD_MODES):
        ax = axes[i]
        df = data_map.get(reward_mode)
        
        if df is not None:
            # Split by partner_helped_last
            df_yes = df[df['partner_helped_last'] == 'Yes'].sort_values('turn')
            df_no = df[df['partner_helped_last'] == 'No'].sort_values('turn')
            
            # Agent lines (solid)
            if len(df_yes) > 0:
                ax.plot(df_yes['turn'], df_yes['agent_helping_rate'], 
                       color=LINE_COLORS['Yes'], linestyle='-', marker='o', label='Agent: Yes')
            if len(df_no) > 0:
                ax.plot(df_no['turn'], df_no['agent_helping_rate'], 
                       color=LINE_COLORS['No'], linestyle='-', marker='s', label='Agent: No')
            
            # Human lines (dashed)
            if len(df_yes) > 0:
                ax.plot(df_yes['turn'], df_yes['human_helping_rate'], 
                       color=LINE_COLORS['Yes'], linestyle='--', marker='o', alpha=0.6)
            if len(df_no) > 0:
                ax.plot(df_no['turn'], df_no['human_helping_rate'], 
                       color=LINE_COLORS['No'], linestyle='--', marker='s', alpha=0.6)
            
            ax.set_xlabel('Turn')
            ax.set_ylabel('Helping Rate')
            ax.set_title(reward_mode.capitalize())
            ax.set_xticks(range(0, 10))
            ax.set_ylim(0, ylim_max)
            
            if i == 0:
                ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(reward_mode.capitalize())
    
    # Human reference panel (dashed lines only)
    ax_human = axes[4]
    human_data = None
    for reward_mode in REWARD_MODES:
        df = data_map.get(reward_mode)
        if df is not None:
            human_data = df
            break
    
    if human_data is not None:
        df_yes = human_data[human_data['partner_helped_last'] == 'Yes'].sort_values('turn')
        df_no = human_data[human_data['partner_helped_last'] == 'No'].sort_values('turn')
        
        if len(df_yes) > 0:
            ax_human.plot(df_yes['turn'], df_yes['human_helping_rate'], 
                         color=LINE_COLORS['Yes'], linestyle='--', marker='o', label='Partner: Yes')
        if len(df_no) > 0:
            ax_human.plot(df_no['turn'], df_no['human_helping_rate'], 
                         color=LINE_COLORS['No'], linestyle='--', marker='s', label='Partner: No')
        
        ax_human.set_xlabel('Turn')
        ax_human.set_ylabel('Helping Rate')
        ax_human.set_title('Human')
        ax_human.set_xticks(range(0, 10))
        ax_human.set_ylim(0, ylim_max)
        ax_human.legend(fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'fig_metric5_reciprocity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def main():
    """Generate all five metric figures."""
    ensure_figures_dir()
    
    print("Generating figures...")
    plot_metric_1()
    plot_metric_2()
    plot_metric_3()
    plot_metric_4()
    plot_metric_5()
    
    print("Done!")


if __name__ == '__main__':
    main()
