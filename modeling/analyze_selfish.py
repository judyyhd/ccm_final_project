"""
Analyze selfish agent reciprocity and behavior patterns.

This script loads the trained selfish agent and runs it on all human game replays,
collecting per-turn behavioral features to understand why the selfish agent exhibits
reciprocity despite having no reciprocity reward shaping.
"""

import os
from contextlib import redirect_stdout
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from farm_env import FarmEnv
from farmgame_io import load_sessions


def load_selfish_model(models_dir='models'):
    """Load the pre-trained selfish agent."""
    model_path = os.path.join(models_dir, 'ppo_selfish.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return PPO.load(model_path)


def run_agent_on_session(env, model, session_df, game_id):
    """
    Run the agent on a single session (human game replay).
    
    Returns a list of per-turn feature dicts.
    """
    turns = []
    obs, _ = env.reset()
    
    done = False
    turn_idx = 0
    
    while not done:
        # Predict action
        try:
            action, _ = model.predict(obs, deterministic=True)
        except:
            break
        
        # Capture pre-action observation features
        # obs format: [agent_pos_x, agent_pos_y, partner_pos_x, partner_pos_y, 
        #              agent_energy, agent_bp_fill, partner_bp_fill, items..., history]
        pre_energy = obs[4] if len(obs) > 4 else 0
        pre_bp_fill = obs[5] if len(obs) > 5 else 0
        
        # Take action
        obs_new, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Determine if agent helped by checking if backpack fill increased
        post_bp_fill = obs_new[5] if len(obs_new) > 5 else 0
        agent_helped = 1 if post_bp_fill > pre_bp_fill else 0
        
        # Get partner helping from history (last element of observation or from partner_helped_history)
        partner_helped_last = 1 if hasattr(env, 'partner_helped_history') and len(env.partner_helped_history) > 0 and env.partner_helped_history[-1] else 0
        
        # Approximate distance from agent and partner positions in obs
        agent_x, agent_y = obs[0], obs[1]
        partner_x, partner_y = obs[2], obs[3]
        distance = ((agent_x - partner_x)**2 + (agent_y - partner_y)**2) ** 0.5 * 25  # Denormalize from [0,1]
        
        # Store turn data
        turns.append({
            'turn': turn_idx,
            'agent_helped': agent_helped,
            'partner_helped_last': partner_helped_last,
            'ownBPfill': pre_bp_fill * 100,  # Convert from [0,1] to percentage
            'ownEnergy': pre_energy,
            'distance_to_partner_veg': distance,
            'partner_needs_help': 1 if post_bp_fill < 0.8 else 0,  # If not close to full
        })
        
        turn_idx += 1
        obs = obs_new
        
        if turn_idx > 100:  # Safety limit
            break
    
    return turns


def analyze_selfish_agent():
    """
    Load selfish agent, run on all human sessions, analyze helping patterns.
    """
    # Load model
    print("Loading selfish agent model...")
    model = load_selfish_model()
    
    # Load human sessions
    print("Loading human sessions...")
    sessions = load_sessions("../data/trialdf.csv", print_progress=False)
    
    # Collect per-turn data
    all_turns = []
    session_count = 0
    
    # Count total games
    total_games = sum(len(games) for games in sessions.values())
    print(f"Running agent on {total_games} games...")
    
    for session_name, games in sessions.items():
        for game_id, game in enumerate(games):
            try:
                # Create environment with human data
                env = FarmEnv(
                    human_game=game,
                    agent_color="red",
                    reward_mode="selfish",
                    history_window=3,
                )
                
                # Run agent on this game
                turns = run_agent_on_session(env, model, pd.DataFrame(game), f"{session_name}_{game_id}")
                all_turns.extend(turns)
                session_count += 1
                
                if session_count % 50 == 0:
                    print(f"  Processed {session_count} games...")
            except Exception as e:
                print(f"  Error processing game {session_name}_{game_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if not all_turns:
        print("ERROR: No turn data collected!")
        return
    
    # Convert to DataFrame
    turns_df = pd.DataFrame(all_turns)
    
    # Save raw data
    os.makedirs('results', exist_ok=True)
    output_path = os.path.join('results', 'selfish_agent_turns.csv')
    turns_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(turns_df)} turns to {output_path}")
    
    # Analyze patterns
    print("\n" + "="*60)
    print("SELFISH AGENT ANALYSIS")
    print("="*60)
    
    # Redirect all print output to a .txt file
    import sys
    output_path = os.path.join(os.path.dirname(__file__), 'selfish_agent_analysis.txt')
    from contextlib import redirect_stderr
    with open(output_path, 'w') as f, redirect_stdout(f), redirect_stderr(f):
        # 1. Helping rate conditional on partner help
        print("\n1. HELPING RATE BY PARTNER'S PREVIOUS TURN:")
        print("-" * 60)
        for partner_helped in [0, 1]:
            label = "Partner Helped" if partner_helped else "Partner Did Not Help"
            subset = turns_df[turns_df['partner_helped_last'] == partner_helped]
            if len(subset) > 0:
                helping_rate = subset['agent_helped'].mean()
                n = len(subset)
                print(f"  {label:30} (n={n:4}): {helping_rate:.3f}")
        
        reciprocity_gap = (turns_df[turns_df['partner_helped_last'] == 1]['agent_helped'].mean() -
                           turns_df[turns_df['partner_helped_last'] == 0]['agent_helped'].mean())
        print(f"\n  Reciprocity Gap: {reciprocity_gap:.4f}")
        print(f"  (Positive = agent helps more when partner helped)")
        
        # 2. State features by helping behavior
        print("\n2. STATE FEATURES BY HELPING BEHAVIOR:")
        print("-" * 60)
        for helped in [0, 1]:
            label = "When Agent Helped" if helped else "When Agent Did Not Help"
            subset = turns_df[turns_df['agent_helped'] == helped]
            if len(subset) > 0:
                print(f"\n  {label} (n={len(subset)}):")
                print(f"    Mean BP Fill:              {subset['ownBPfill'].mean():.2f}%")
                print(f"    Mean Energy:              {subset['ownEnergy'].mean():.2f}")
                print(f"    Mean Distance to Partner: {subset['distance_to_partner_veg'].mean():.2f}")
                print(f"    Partner Needs Help:       {subset['partner_needs_help'].mean():.3f}")
        
        # 3. Generate figures
        print("\n3. Generating figures...")
        _plot_helping_by_state(turns_df)
        _plot_reciprocity_comparison(turns_df)
        
        print("\nDone!")


def _plot_helping_by_state(turns_df):
    """Generate 2x3 panel figure of helping rates by state features."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Selfish Agent: Helping Rate by State Features', fontsize=14, fontweight='bold')
    
    # Panel 1: By BP fill quartiles
    ax = axes[0, 0]
    bpfill_unique = turns_df['ownBPfill'].nunique(dropna=True)
    try:
        if bpfill_unique >= 4:
            turns_df['bp_quartile'] = pd.qcut(turns_df['ownBPfill'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
        else:
            turns_df['bp_quartile'] = pd.cut(turns_df['ownBPfill'], bins=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    except ValueError:
        # Fallback: assign all to a single bin
        turns_df['bp_quartile'] = 'Q1 (Low)'
    helping_by_bp = turns_df.groupby('bp_quartile')['agent_helped'].agg(['mean', 'count'])
    ax.bar(range(len(helping_by_bp)), helping_by_bp['mean'], color='steelblue')
    ax.set_xticks(range(len(helping_by_bp)))
    ax.set_xticklabels(helping_by_bp.index, rotation=45, ha='right')
    ax.set_ylabel('Helping Rate')
    ax.set_xlabel('Backpack Fill Quartile')
    ax.set_ylim(0, 1)
    ax.set_title('Helping by Own Backpack Fill')
    
    # Panel 2: By energy quartiles
    ax = axes[0, 1]
    energy_unique = turns_df['ownEnergy'].nunique(dropna=True)
    try:
        if energy_unique >= 4:
            turns_df['energy_quartile'] = pd.qcut(turns_df['ownEnergy'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
        else:
            turns_df['energy_quartile'] = pd.cut(turns_df['ownEnergy'], bins=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    except ValueError:
        turns_df['energy_quartile'] = 'Q1 (Low)'
    helping_by_energy = turns_df.groupby('energy_quartile')['agent_helped'].agg(['mean', 'count'])
    ax.bar(range(len(helping_by_energy)), helping_by_energy['mean'], color='coral')
    ax.set_xticks(range(len(helping_by_energy)))
    ax.set_xticklabels(helping_by_energy.index, rotation=45, ha='right')
    ax.set_ylabel('Helping Rate')
    ax.set_xlabel('Energy Quartile')
    ax.set_ylim(0, 1)
    ax.set_title('Helping by Own Energy')
    
    # Panel 3: By distance to partner veg
    ax = axes[0, 2]
    turns_df['distance_bin'] = pd.cut(turns_df['distance_to_partner_veg'], 
                                       bins=[0, 5, 10, 15, 100], 
                                       labels=['0-5', '6-10', '11-15', '>15'])
    helping_by_dist = turns_df.groupby('distance_bin')['agent_helped'].agg(['mean', 'count'])
    ax.bar(range(len(helping_by_dist)), helping_by_dist['mean'], color='mediumseagreen')
    ax.set_xticks(range(len(helping_by_dist)))
    ax.set_xticklabels(helping_by_dist.index, rotation=45, ha='right')
    ax.set_ylabel('Helping Rate')
    ax.set_xlabel('Distance to Partner Veg')
    ax.set_ylim(0, 1)
    ax.set_title('Helping by Distance to Partner Vegetables')
    
    # Panel 4: By partner need
    ax = axes[1, 0]
    helping_by_partner_need = turns_df.groupby('partner_needs_help')['agent_helped'].agg(['mean', 'count'])
    labels = ['Partner Full', 'Partner Needs Help']
    ax.bar(['No', 'Yes'], helping_by_partner_need['mean'], color='mediumpurple')
    ax.set_ylabel('Helping Rate')
    ax.set_xlabel('Partner Needs Help')
    ax.set_ylim(0, 1)
    ax.set_title('Helping by Partner\'s Need')
    
    # Panel 5: By turn number
    ax = axes[1, 1]
    helping_by_turn = turns_df.groupby('turn')['agent_helped'].agg(['mean', 'count'])
    ax.plot(helping_by_turn.index, helping_by_turn['mean'], marker='o', color='darkred', linewidth=2)
    ax.set_ylabel('Helping Rate')
    ax.set_xlabel('Turn')
    ax.set_ylim(0, 1)
    ax.set_title('Helping by Turn Number')
    
    # Panel 6: Reciprocity (key finding)
    ax = axes[1, 2]
    reciprocity_data = turns_df.groupby('partner_helped_last')['agent_helped'].agg(['mean', 'count'])
    labels = ['Partner\nDid Not Help', 'Partner\nHelped']
    colors = ['#FF6B6B', '#51CF66']
    bars = ax.bar(labels, reciprocity_data['mean'], color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Agent Helping Rate', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Reciprocity Effect (KEY FINDING)', fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', 'selfish_helping_by_state.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()


def _plot_reciprocity_comparison(turns_df):
    """Generate figure comparing helping rates when partner helped vs. didn't."""
    # Compute helping rates for each condition
    subset_helped = turns_df[turns_df['partner_helped_last'] == 1]['agent_helped']
    subset_not_helped = turns_df[turns_df['partner_helped_last'] == 0]['agent_helped']
    
    helped_rate = subset_helped.mean() if len(subset_helped) > 0 else 0
    not_helped_rate = subset_not_helped.mean() if len(subset_not_helped) > 0 else 0
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    conditions = ['Partner\nHelped\nLast Turn', 'Partner\nDid NOT\nHelp Last Turn']
    helping_rates = [helped_rate, not_helped_rate]
    colors = ['#51CF66', '#FF6B6B']
    
    # Create bars
    bars = ax.bar(conditions, helping_rates, color=colors, edgecolor='black', linewidth=2, width=0.5)
    
    # Formatting
    ax.set_ylabel('Agent Helping Rate', fontweight='bold', fontsize=12)
    ax.set_title('Selfish Agent: Helping Behavior by Partner\'s Previous Action', 
                 fontweight='bold', fontsize=13)
    ax.set_ylim(0, max(helping_rates) * 1.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, helping_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add sample sizes as text below
    n_helped = len(subset_helped)
    n_not_helped = len(subset_not_helped)
    ax.text(0, -0.15, f'n={n_helped}', ha='center', transform=ax.get_xaxis_transform(), fontsize=10)
    ax.text(1, -0.15, f'n={n_not_helped}', ha='center', transform=ax.get_xaxis_transform(), fontsize=10)
    
    # Add horizontal line at mean helping rate
    overall_mean = turns_df['agent_helped'].mean()
    ax.axhline(overall_mean, color='gray', linestyle='--', linewidth=2, alpha=0.7, label=f'Overall Mean: {overall_mean:.1%}')
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    output_path = os.path.join('figures', 'selfish_reciprocity_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    analyze_selfish_agent()
