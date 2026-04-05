"""
Evaluation script for trained PPO agents.
Computes behavioral metrics and compares to human data.
"""

import os
import csv
import sys
import logging
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import farmgame
import farmgame_io
from farm_env import FarmEnv
import utils

# Configure logging
log_file = "evaluation_output.txt"
# Clear the log file on startup
with open(log_file, "w") as f:
    f.write("")
    
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _build_patch_uniformity_map(trialdf_path: str) -> dict:
    """
    Build mapping from objectLayer to patchUniformity from trialdf.csv.
    
    Args:
        trialdf_path: Path to trialdf.csv file
        
    Returns:
        Dict mapping objectLayer -> patchUniformity
    """
    df = pd.read_csv(trialdf_path)
    
    # Drop rows with NaN in objectLayer or patchUniformity
    df = df.dropna(subset=["objectLayer", "patchUniformity"])
    
    # Keep only first row per unique objectLayer (should be consistent within same objectLayer)
    df = df.drop_duplicates(subset=["objectLayer"], keep="first")
    
    # Build mapping
    patch_map = dict(zip(df["objectLayer"], df["patchUniformity"]))
    
    # Add fallback for any unknown layers
    if not patch_map:
        logger.warning("No valid patchUniformity mappings found in trialdf.csv")
        patch_map = {}
    
    return patch_map


def run_agent_episode(agent, human_game, agent_color="red", reward_mode="selfish"):
    """
    Run a trained agent through a game and collect behavioral metrics.

    Returns a list of dicts, one per turn of the agent.
    """
    env = FarmEnv(
        human_game=human_game,
        agent_color=agent_color,
        reward_mode=reward_mode,
        history_window=3,
    )
    obs, _ = env.reset()

    metrics_list = []
    turn_number = 0

    while True:
        # Get agent's action
        action_idx, _ = agent.predict(obs, deterministic=True)

        # Get legal actions to extract the actual action
        legal_actions = env.state.legal_actions()
        if action_idx >= len(legal_actions):
            action_idx = len(legal_actions) - 1
        agent_action = legal_actions[action_idx]

        # Determine if this is a helping action
        is_helping = (
            agent_action.type == farmgame.ActionType.veggie
            and agent_action.color != agent_color
        )

        # Get agent state before action
        agent_state = env.state.redplayer if agent_color == "red" else env.state.purpleplayer
        partner_color = "purple" if agent_color == "red" else "red"

        # Compute metrics for this turn
        metrics = {
            "turn": turn_number,
            "agent_color": agent_color,
            "helping_event": int(is_helping),
            "ownBPsize": agent_state["backpack"]["capacity"],
            "ownEnergy": agent_state["energy"],
            "ownDistanceToClosestOtherVeg": compute_distance_to_closest_other_veg(
                env.state, agent_color
            ),
            "objectLayer": env.state.objectLayer,
        }

        # Compute partner helped last turn (from history)
        partner_helped_lastturn = (
            env.partner_helped_history[0] if len(env.partner_helped_history) > 0 else False
        )
        metrics["partner_helped_lasttrial"] = int(partner_helped_lastturn)

        metrics_list.append(metrics)

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action_idx)
        turn_number += 1

        if terminated:
            break

    return metrics_list


def compute_distance_to_closest_other_veg(state, agent_color):
    """Compute Manhattan distance to closest partner vegetable on farm."""
    agent = state.redplayer if agent_color == "red" else state.purpleplayer
    partner_color = "purple" if agent_color == "red" else "red"
    partner_veggies = [
        item for item in state.items if item.color == partner_color and item.status == "farm"
    ]

    if not partner_veggies:
        return float("inf")

    return min(
        utils.getManhattanDistance(agent["loc"], veg.loc) for veg in partner_veggies
    )


def load_human_metrics(filepath: str, agent_color: str = "red"):
    """
    Load human behavioral metrics from trialdf.csv.
    Filter to only rows where agent (player) is agent_color.
    """
    df = pd.read_csv(filepath)

    # Map subjid to agent color (red or purple) based on suffix
    df["agent_color"] = df["subjid"].str.extract(r"(red|purple)$")

    # Filter to specified agent color
    df = df[df["agent_color"] == agent_color].reset_index(drop=True)
    
    # Filter to only non-gameover rows
    df = df[df["gameover"] == False].reset_index(drop=True)

    # Extract or compute metrics
    metrics_list = []
    for _, row in df.iterrows():
        try:
            # Safely get columns, with fallback to defaults
            helping_event = int(row.get("helping_event", 0)) if pd.notna(row.get("helping_event")) else 0
            own_bp_size = int(row.get("ownBPsize", 4)) if pd.notna(row.get("ownBPsize")) else 4
            own_energy = int(row.get("ownEnergy", 50)) if pd.notna(row.get("ownEnergy")) else 50
            own_distance = int(row.get("ownDistanceToClosestOtherVeg", 0)) if pd.notna(row.get("ownDistanceToClosestOtherVeg")) else 0
            partner_helped = int(row.get("partner_helped_lasttrial", 0)) if pd.notna(row.get("partner_helped_lasttrial")) else 0
            patch_uniformity = row.get("patchUniformity", "unknown") if pd.notna(row.get("patchUniformity")) else "unknown"
            turn_count = int(row.get("turnCount", 0)) if pd.notna(row.get("turnCount")) else 0
            
            metrics = {
                "helping_event": helping_event,
                "ownBPsize": own_bp_size,
                "ownEnergy": own_energy,
                "ownDistanceToClosestOtherVeg": own_distance,
                "partner_helped_lasttrial": partner_helped,
                "patchUniformity": patch_uniformity,
                "turnCount": turn_count,
            }
            metrics_list.append(metrics)
        except Exception as e:
            continue

    return metrics_list


def compute_metric_1_backpack_size(agent_metrics_list, human_metrics_list):
    """Helpfulness by backpack size."""
    if not agent_metrics_list or not human_metrics_list:
        return {"backpack_size": [], "agent_helping_rate": [], "human_helping_rate": []}
    
    agent_df = pd.DataFrame(agent_metrics_list)
    human_df = pd.DataFrame(human_metrics_list)

    result = {"backpack_size": [], "agent_helping_rate": [], "human_helping_rate": []}

    for bp_size in [3, 4, 5]:
        agent_subset = agent_df[agent_df["ownBPsize"] == bp_size]
        human_subset = human_df[human_df["ownBPsize"] == bp_size]

        if len(agent_subset) > 0 and len(human_subset) > 0:
            agent_rate = agent_subset["helping_event"].mean()
            human_rate = human_subset["helping_event"].mean()

            result["backpack_size"].append(bp_size)
            result["agent_helping_rate"].append(agent_rate)
            result["human_helping_rate"].append(human_rate)

    return result


def compute_metric_2_patch_uniformity(agent_metrics_list, human_metrics_list, patch_uniformity_map: dict):
    """Helpfulness by patch uniformity.
    
    Args:
        agent_metrics_list: List of agent metrics dicts with "objectLayer" field
        human_metrics_list: List of human metrics dicts with "patchUniformity" field
        patch_uniformity_map: Dict mapping objectLayer -> patchUniformity
    
    Returns:
        Dict with patchUniformity categories and helping rates for agents vs humans
    """
    if not agent_metrics_list or not human_metrics_list:
        return {
            "patchUniformity": [],
            "agent_helping_rate": [],
            "human_helping_rate": [],
        }
    
    # For each agent metric, look up patchUniformity from objectLayer
    agent_df = pd.DataFrame(agent_metrics_list)
    agent_df["patchUniformity"] = agent_df["objectLayer"].map(patch_uniformity_map).fillna("unknown")
    
    human_df = pd.DataFrame(human_metrics_list)
    
    result = {
        "patchUniformity": [],
        "agent_helping_rate": [],
        "human_helping_rate": [],
    }
    
    # Compute for each patchUniformity value found in the agent data
    for patch_type in sorted(agent_df["patchUniformity"].unique()):
        agent_subset = agent_df[agent_df["patchUniformity"] == patch_type]
        human_subset = human_df[human_df["patchUniformity"] == patch_type]
        
        if len(agent_subset) > 0 and len(human_subset) > 0:
            agent_rate = agent_subset["helping_event"].mean()
            human_rate = human_subset["helping_event"].mean()
            
            result["patchUniformity"].append(patch_type)
            result["agent_helping_rate"].append(agent_rate)
            result["human_helping_rate"].append(human_rate)
    
    return result


def compute_metric_3_distance_to_partner_veg(agent_metrics_list, human_metrics_list):
    """Helping rate by distance to nearest partner vegetable."""
    if not agent_metrics_list or not human_metrics_list:
        return {"distance_bin": [], "agent_helping_rate": [], "human_helping_rate": []}
    
    agent_df = pd.DataFrame(agent_metrics_list)
    human_df = pd.DataFrame(human_metrics_list)

    result = {"distance_bin": [], "agent_helping_rate": [], "human_helping_rate": []}

    # Distance bins: [0-2, 3-5, 6-10, 11+]
    bins = [(0, 2), (3, 5), (6, 10), (11, float("inf"))]
    bin_labels = ["0-2", "3-5", "6-10", "11+"]

    for (low, high), label in zip(bins, bin_labels):
        agent_subset = agent_df[
            (agent_df["ownDistanceToClosestOtherVeg"] >= low)
            & (agent_df["ownDistanceToClosestOtherVeg"] <= high)
        ]
        human_subset = human_df[
            (human_df["ownDistanceToClosestOtherVeg"] >= low)
            & (human_df["ownDistanceToClosestOtherVeg"] <= high)
        ]

        if len(agent_subset) > 0 and len(human_subset) > 0:
            agent_rate = agent_subset["helping_event"].mean()
            human_rate = human_subset["helping_event"].mean()

            result["distance_bin"].append(label)
            result["agent_helping_rate"].append(agent_rate)
            result["human_helping_rate"].append(human_rate)

    return result


def compute_metric_4_remaining_energy(agent_metrics_list, human_metrics_list):
    """Helping rate by remaining energy (deciles)."""
    if not agent_metrics_list or not human_metrics_list:
        return {"energy_bin": [], "agent_helping_rate": [], "human_helping_rate": []}
    
    agent_df = pd.DataFrame(agent_metrics_list)
    human_df = pd.DataFrame(human_metrics_list)

    result = {"energy_bin": [], "agent_helping_rate": [], "human_helping_rate": []}

    # Energy deciles: 0-10, 11-20, ..., 91-100
    bins = [(i, i + 10) for i in range(0, 100, 10)]
    bin_labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]

    for (low, high), label in zip(bins, bin_labels):
        agent_subset = agent_df[(agent_df["ownEnergy"] >= low) & (agent_df["ownEnergy"] < high)]
        human_subset = human_df[(human_df["ownEnergy"] >= low) & (human_df["ownEnergy"] < high)]

        if len(agent_subset) > 0 and len(human_subset) > 0:
            agent_rate = agent_subset["helping_event"].mean()
            human_rate = human_subset["helping_event"].mean()

            result["energy_bin"].append(label)
            result["agent_helping_rate"].append(agent_rate)
            result["human_helping_rate"].append(human_rate)

    return result


def compute_metric_5_conditional_on_partner_help(agent_metrics_list, human_metrics_list):
    """Helping rate by partner's action in previous turn (conditional), broken down per turn."""
    if not agent_metrics_list or not human_metrics_list:
        return {
            "turn": [],
            "partner_helped_last": [],
            "agent_helping_rate": [],
            "human_helping_rate": [],
        }
    
    agent_df = pd.DataFrame(agent_metrics_list)
    human_df = pd.DataFrame(human_metrics_list)

    result = {
        "turn": [],
        "partner_helped_last": [],
        "agent_helping_rate": [],
        "human_helping_rate": [],
    }

    # Compute for each turn (0-9) and partner_helped_last condition
    for turn in range(10):
        for partner_helped in [False, True]:
            agent_subset = agent_df[
                (agent_df["turn"] == turn) & 
                (agent_df["partner_helped_lasttrial"] == (1 if partner_helped else 0))
            ]
            human_subset = human_df[
                (human_df["turnCount"] == turn) & 
                (human_df["partner_helped_lasttrial"] == (1 if partner_helped else 0))
            ]

            if len(agent_subset) > 0 and len(human_subset) > 0:
                agent_rate = agent_subset["helping_event"].mean()
                human_rate = human_subset["helping_event"].mean()

                label = "Yes" if partner_helped else "No"
                result["turn"].append(turn)
                result["partner_helped_last"].append(label)
                result["agent_helping_rate"].append(agent_rate)
                result["human_helping_rate"].append(human_rate)

    return result


def evaluate_agent(reward_mode: str, output_dir: str = "models"):
    """
    Evaluate a trained agent and compute metrics.

    Args:
        reward_mode: One of "selfish", "capacity", "proximity", "reciprocity".
        output_dir: Directory where the model is saved.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating PPO agent with reward_mode='{reward_mode}'")
    logger.info(f"{'='*60}")

    # Load the trained model
    model_path = os.path.join(output_dir, f"ppo_{reward_mode}")
    try:
        agent = PPO.load(model_path)
        logger.info(f"Loaded model from {model_path}.zip")
    except Exception as e:
        logger.info(f"Error loading model: {e}")
        return

    # Load human sessions
    logger.info("Loading human sessions for evaluation...")
    sessions = farmgame_io.load_sessions("../data/trialdf.csv", print_progress=False)

    # Collect metrics from agent
    agent_metrics_all = []
    num_games = 0
    for session_name, session in sessions.items():
        for game in session:
            try:
                metrics = run_agent_episode(agent, game, agent_color="red", reward_mode=reward_mode)
                agent_metrics_all.extend(metrics)
                num_games += 1
            except Exception as e:
                logger.info(f"Error evaluating game: {e}")
                continue

    logger.info(f"Evaluated agent on {num_games} games.")

    # Load human metrics
    logger.info("Loading human metrics from trialdf.csv...")
    human_metrics_all = load_human_metrics("../data/trialdf.csv", agent_color="red")

    # Build patch uniformity mapping
    logger.info("Building patch uniformity mapping...")
    patch_uniformity_map = _build_patch_uniformity_map("../data/trialdf.csv")

    # Compute all metrics
    metric_1 = compute_metric_1_backpack_size(agent_metrics_all, human_metrics_all)
    metric_2 = compute_metric_2_patch_uniformity(agent_metrics_all, human_metrics_all, patch_uniformity_map)
    metric_3 = compute_metric_3_distance_to_partner_veg(agent_metrics_all, human_metrics_all)
    metric_4 = compute_metric_4_remaining_energy(agent_metrics_all, human_metrics_all)
    metric_5 = compute_metric_5_conditional_on_partner_help(agent_metrics_all, human_metrics_all)

    # Save results to CSV
    os.makedirs("results", exist_ok=True)

    # Metric 1: Backpack Size
    if metric_1["backpack_size"]:
        df1 = pd.DataFrame(metric_1)
        df1.to_csv(f"results/metrics_{reward_mode}_metric1_backpack.csv", index=False)
        logger.info(f"Saved metric 1 to results/metrics_{reward_mode}_metric1_backpack.csv")

    # Metric 2: Patch Uniformity
    if metric_2["patchUniformity"]:
        df2 = pd.DataFrame(metric_2)
        df2.to_csv(f"results/metrics_{reward_mode}_metric2_patchuniformity.csv", index=False)
        logger.info(f"Saved metric 2 to results/metrics_{reward_mode}_metric2_patchuniformity.csv")

    # Metric 3: Distance
    if metric_3["distance_bin"]:
        df3 = pd.DataFrame(metric_3)
        df3.to_csv(f"results/metrics_{reward_mode}_metric3_distance.csv", index=False)
        logger.info(f"Saved metric 3 to results/metrics_{reward_mode}_metric3_distance.csv")

    # Metric 4: Energy
    if metric_4["energy_bin"]:
        df4 = pd.DataFrame(metric_4)
        df4.to_csv(f"results/metrics_{reward_mode}_metric4_energy.csv", index=False)
        logger.info(f"Saved metric 4 to results/metrics_{reward_mode}_metric4_energy.csv")

    # Metric 5: Partner Reciprocity
    if metric_5["partner_helped_last"]:
        df5 = pd.DataFrame(metric_5)
        df5.to_csv(f"results/metrics_{reward_mode}_metric5_reciprocity.csv", index=False)
        logger.info(f"Saved metric 5 to results/metrics_{reward_mode}_metric5_reciprocity.csv")

    # Print summary
    logger.info("\n" + "-" * 60)
    logger.info(f"Summary for reward_mode='{reward_mode}':")
    logger.info("-" * 60)

    if agent_metrics_all:
        agent_help_rate = np.mean([m["helping_event"] for m in agent_metrics_all])
    else:
        agent_help_rate = 0.0

    if human_metrics_all:
        human_help_rate = np.mean([m["helping_event"] for m in human_metrics_all])
    else:
        human_help_rate = 0.0

    logger.info(f"Agent overall helping rate: {agent_help_rate:.3f}")
    logger.info(f"Human overall helping rate: {human_help_rate:.3f}")
    logger.info(f"Difference: {agent_help_rate - human_help_rate:.3f}")


def main():
    """Evaluate all trained agents."""
    reward_modes = ["selfish", "capacity", "proximity", "reciprocity"]

    for reward_mode in reward_modes:
        try:
            evaluate_agent(reward_mode)
        except Exception as e:
            logger.info(f"Error evaluating agent with reward_mode='{reward_mode}': {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "="*60)
    logger.info("Evaluation complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
