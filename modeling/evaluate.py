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
log_file = os.path.join(os.path.dirname(__file__), "evaluation_output.txt")
    
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _build_patch_uniformity_map(env_features_path: str = None) -> dict:
    """
    Build mapping from objectLayer to patchUniformity from environment_features.csv.
    
    Args:
        env_features_path: Path to environment_features.csv file
        
    Returns:
        Dict mapping objectLayer -> patchUniformity
    """
    if env_features_path is None:
        env_features_path = "../data/environment_features.csv"
    
    # If path is relative, resolve it relative to this file's location
    if not os.path.isabs(env_features_path):
        env_features_path = os.path.join(os.path.dirname(__file__), env_features_path)
    
    logger.info(f"Loading patch uniformity from {env_features_path}")
    
    if not os.path.exists(env_features_path):
        logger.warning(f"File not found: {env_features_path}")
        return {}
    
    df = pd.read_csv(env_features_path)
    
    # Check if required columns exist
    if "objectLayer" not in df.columns or "patchUniformity" not in df.columns:
        logger.warning("Columns 'objectLayer' or 'patchUniformity' not found in environment_features.csv")
        return {}
    
    # Drop rows with NaN in objectLayer or patchUniformity
    df = df.dropna(subset=["objectLayer", "patchUniformity"])
    
    # Keep only first row per unique objectLayer (should be consistent within same objectLayer)
    df = df.drop_duplicates(subset=["objectLayer"], keep="first")
    
    # Build mapping - convert patchUniformity to string for consistency
    patch_map = dict(zip(df["objectLayer"], df["patchUniformity"].astype(str)))
    
    if not patch_map:
        logger.warning("No valid patchUniformity mappings found in environment_features.csv")
    else:
        logger.info(f"Loaded {len(patch_map)} patch uniformity mappings")
    
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
    Merge in patchUniformity from environment_features.csv.
    """
    # If filepath is relative, resolve it relative to this file's location
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.path.dirname(__file__), filepath)
    
    df = pd.read_csv(filepath)

    # Load environment features and merge on objectLayer
    env_features_path = os.path.join(os.path.dirname(__file__), "../data/environment_features.csv")
    if os.path.exists(env_features_path):
        env_df = pd.read_csv(env_features_path)[["objectLayer", "patchUniformity"]]
        df = df.merge(env_df, on="objectLayer", how="left")
    
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
                "patchUniformity": str(patch_uniformity),  # Convert to string for consistency
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
    """Helpfulness by patch uniformity (True=Uniform, False=Non-uniform)."""
    if not agent_metrics_list or not human_metrics_list:
        return {
            "patchUniformity": [],
            "agent_helping_rate": [],
            "human_helping_rate": [],
        }
    
    # If patch_uniformity_map is empty, warn and return empty result
    if not patch_uniformity_map:
        logger.warning("Patch uniformity map is empty; skipping metric 2 computation")
        return {
            "patchUniformity": [],
            "agent_helping_rate": [],
            "human_helping_rate": [],
        }

    agent_df = pd.DataFrame(agent_metrics_list)
    agent_df["patchUniformity"] = (
        agent_df["objectLayer"].map(patch_uniformity_map).astype(str)
    )

    human_df = pd.DataFrame(human_metrics_list)
    human_df["patchUniformity"] = human_df["patchUniformity"].astype(str)

    result = {
        "patchUniformity": [],
        "agent_helping_rate": [],
        "human_helping_rate": [],
    }

    # "True" = uniform patches, "False" = non-uniform patches
    for val, label in [("True", "Uniform"), ("False", "Non-uniform")]:
        agent_subset = agent_df[agent_df["patchUniformity"] == val]
        human_subset = human_df[human_df["patchUniformity"] == val]

        if len(agent_subset) > 0 and len(human_subset) > 0:
            result["patchUniformity"].append(label)
            result["agent_helping_rate"].append(agent_subset["helping_event"].mean())
            result["human_helping_rate"].append(human_subset["helping_event"].mean())

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


def evaluate_agent(reward_mode: str, output_dir: str = "models", tag: str = ""):
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
    suffix = f"_{tag}" if tag else ""
    model_path = os.path.join(output_dir, f"ppo_{reward_mode}{suffix}")
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
    patch_uniformity_map = _build_patch_uniformity_map()

    # Compute all metrics
    metric_1 = compute_metric_1_backpack_size(agent_metrics_all, human_metrics_all)
    metric_2 = compute_metric_2_patch_uniformity(agent_metrics_all, human_metrics_all, patch_uniformity_map)
    metric_3 = compute_metric_3_distance_to_partner_veg(agent_metrics_all, human_metrics_all)
    metric_4 = compute_metric_4_remaining_energy(agent_metrics_all, human_metrics_all)
    metric_5 = compute_metric_5_conditional_on_partner_help(agent_metrics_all, human_metrics_all)

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    result_prefix = f"results/metrics_{reward_mode}{suffix}"

    # Metric 1: Backpack Size
    if metric_1["backpack_size"]:
        df1 = pd.DataFrame(metric_1)
        df1.to_csv(f"{result_prefix}_metric1_backpack.csv", index=False)
        logger.info(f"Saved metric 1 to {result_prefix}_metric1_backpack.csv")

    # Metric 2: Patch Uniformity
    if metric_2["patchUniformity"]:
        df2 = pd.DataFrame(metric_2)
        df2.to_csv(f"{result_prefix}_metric2_patchuniformity.csv", index=False)
        logger.info(f"Saved metric 2 to {result_prefix}_metric2_patchuniformity.csv")

    # Metric 3: Distance
    if metric_3["distance_bin"]:
        df3 = pd.DataFrame(metric_3)
        df3.to_csv(f"{result_prefix}_metric3_distance.csv", index=False)
        logger.info(f"Saved metric 3 to {result_prefix}_metric3_distance.csv")

    # Metric 4: Energy
    if metric_4["energy_bin"]:
        df4 = pd.DataFrame(metric_4)
        df4.to_csv(f"{result_prefix}_metric4_energy.csv", index=False)
        logger.info(f"Saved metric 4 to {result_prefix}_metric4_energy.csv")

    # Metric 5: Partner Reciprocity
    if metric_5["partner_helped_last"]:
        df5 = pd.DataFrame(metric_5)
        df5.to_csv(f"{result_prefix}_metric5_reciprocity.csv", index=False)
        logger.info(f"Saved metric 5 to {result_prefix}_metric5_reciprocity.csv")

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
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agents")
    parser.add_argument("--mode", type=str, default="all",
        choices=["all", "selfish", "capacity", "proximity", "reciprocity", "capacity_proximity"],
        help="Which reward mode to evaluate (default: all)")
    parser.add_argument("--tag", type=str, default="",
        help="Model tag, e.g. 'v2' loads ppo_capacity_v2.zip (default: none)")
    parser.add_argument("--output_dir", type=str, default="models",
        help="Directory where models are saved (default: models)")
    args = parser.parse_args()

    # Clear the log file at the start of this run
    with open(log_file, "w") as f:
        f.write("")
    
    logger.info("="*60)
    logger.info("EVALUATION RUN STARTED")
    logger.info("="*60)

    modes = (
        ["selfish", "capacity", "proximity", "reciprocity", "capacity_proximity"]
        if args.mode == "all" else [args.mode]
    )

    for reward_mode in modes:
        try:
            evaluate_agent(reward_mode, output_dir=args.output_dir, tag=args.tag)
        except Exception as e:
            logger.info(f"Error evaluating agent with reward_mode='{reward_mode}': {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "="*60)
    logger.info("Evaluation complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
