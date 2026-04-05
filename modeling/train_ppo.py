"""
Training script for PPO agents in the farming game environment.
Trains separate agents for each reward mode.
"""

import os
import sys
import logging
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import farmgame_io
from farm_env import FarmEnv

# Configure logging
log_file = os.path.join(os.path.dirname(__file__), "training_output.txt")
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


def make_env(human_game, agent_color="red", reward_mode="selfish"):
    """Factory function to create a FarmEnv."""
    def _init():
        return FarmEnv(
            human_game=human_game,
            agent_color=agent_color,
            reward_mode=reward_mode,
            history_window=3,
        )
    return _init


def train_ppo_agent(reward_mode: str, output_dir: str = "models"):
    """
    Train a PPO agent for a given reward mode.

    Args:
        reward_mode: One of "selfish", "capacity", "proximity", "reciprocity".
        output_dir: Directory to save the trained model.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training PPO agent with reward_mode='{reward_mode}'")
    logger.info(f"{'='*60}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load human sessions
    logger.info("Loading human sessions from data/trialdf.csv...")
    sessions = farmgame_io.load_sessions("../data/trialdf.csv", print_progress=True)

    # Flatten sessions into individual games
    all_games = []
    for session_name, session in sessions.items():
        for game in session:
            all_games.append(game)

    logger.info(f"Loaded {len(all_games)} games from {len(sessions)} sessions.")

    # Create a pool of environments (one per human game)
    # Use SubprocVecEnv for parallel training, or DummyVecEnv for debugging
    num_envs = min(len(all_games), 4)  # Limit to 4 parallel envs
    logger.info(f"Creating {num_envs} vectorized environments...")

    env_fns = [
        make_env(all_games[i % len(all_games)], agent_color="red", reward_mode=reward_mode)
        for i in range(num_envs)
    ]

    # Try SubprocVecEnv, fall back to DummyVecEnv if issues
    try:
        vec_env = SubprocVecEnv(env_fns)
    except Exception as e:
        logger.info(f"SubprocVecEnv failed: {e}. Using DummyVecEnv instead.")
        vec_env = DummyVecEnv(env_fns)

    # Create and train PPO agent
    logger.info("\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
    )

    logger.info(f"Training PPO agent for 500,000 timesteps...")
    
    # Redirect stdout/stderr to capture all training output
    with open(log_file, "a") as f:
        with redirect_stdout(f), redirect_stderr(f):
            model.learn(total_timesteps=500_000)

    # Save the model
    model_path = os.path.join(output_dir, f"ppo_{reward_mode}")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}.zip")

    # Clean up
    vec_env.close()

    return model_path


def main():
    """Train agents for all reward modes."""
    reward_modes = ["selfish", "capacity", "proximity", "reciprocity"]

    for reward_mode in reward_modes:
        try:
            train_ppo_agent(reward_mode)
        except Exception as e:
            logger.info(f"Error training agent with reward_mode='{reward_mode}': {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
