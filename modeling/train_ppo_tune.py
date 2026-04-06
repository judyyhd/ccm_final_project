"""
Training script for PPO agents in the farming game environment.
Supports command-line hyperparameter tuning.

Usage:
    # Train all modes with defaults
    python train_ppo_tune.py

    # Train one mode with custom hyperparameters
    python train_ppo_tune.py --mode capacity --timesteps 1000000 --num_envs 16

    # Train all modes and save as v2
    python train_ppo_tune.py --timesteps 1000000 --num_envs 16 --tag v2

    # Full custom example
    python train_ppo_tune.py --mode reciprocity --timesteps 1000000 --num_envs 16 --n_steps 1024 --batch_size 128 --ent_coef 0.05 --tag exp1
"""

import os
import sys
import argparse
import logging
from contextlib import redirect_stdout, redirect_stderr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import farmgame_io
from farm_env import FarmEnv

log_file = os.path.join(os.path.dirname(__file__), "training_output.txt")

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
    def _init():
        return FarmEnv(
            human_game=human_game,
            agent_color=agent_color,
            reward_mode=reward_mode,
            history_window=3,
        )
    return _init


def train_ppo_agent(
    reward_mode,
    output_dir="models",
    timesteps=500_000,
    num_envs=4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    tag="",
):
    logger.info(f"\n{'='*60}")
    logger.info(f"Training PPO agent with reward_mode='{reward_mode}'")
    logger.info(f"  timesteps={timesteps:,}, num_envs={num_envs}")
    logger.info(f"  n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}")
    logger.info(f"  gamma={gamma}, ent_coef={ent_coef}, tag='{tag}'")
    logger.info(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading human sessions from data/trialdf.csv...")
    sessions = farmgame_io.load_sessions("../data/trialdf.csv", print_progress=True)
    all_games = [game for session in sessions.values() for game in session]
    logger.info(f"Loaded {len(all_games)} games from {len(sessions)} sessions.")

    n_envs = min(len(all_games), num_envs)
    logger.info(f"Creating {n_envs} vectorized environments...")

    env_fns = [
        make_env(all_games[i % len(all_games)], agent_color="red", reward_mode=reward_mode)
        for i in range(n_envs)
    ]

    try:
        vec_env = SubprocVecEnv(env_fns)
    except Exception as e:
        logger.info(f"SubprocVecEnv failed: {e}. Using DummyVecEnv instead.")
        vec_env = DummyVecEnv(env_fns)

    logger.info("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=1,
    )

    logger.info(f"Training for {timesteps:,} timesteps...")
    with open(log_file, "a") as f:
        with redirect_stdout(f), redirect_stderr(f):
            model.learn(total_timesteps=timesteps)

    suffix = f"_{tag}" if tag else ""
    model_path = os.path.join(output_dir, f"ppo_{reward_mode}{suffix}")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}.zip")

    vec_env.close()
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train PPO agents for farming game")
    parser.add_argument("--mode", type=str, default="all",
        choices=["all", "selfish", "capacity", "proximity", "reciprocity", "capacity_proximity"],
        help="Which reward mode to train (default: all)")
    parser.add_argument("--timesteps",  type=int,   default=500_000, help="Training timesteps (default: 500000)")
    parser.add_argument("--num_envs",   type=int,   default=4,       help="Parallel environments (default: 4)")
    parser.add_argument("--n_steps",    type=int,   default=512,     help="PPO n_steps (default: 512)")
    parser.add_argument("--batch_size", type=int,   default=64,      help="PPO batch size (default: 64)")
    parser.add_argument("--n_epochs",   type=int,   default=10,      help="PPO n_epochs (default: 10)")
    parser.add_argument("--gamma",      type=float, default=0.99,    help="Discount factor (default: 0.99)")
    parser.add_argument("--ent_coef",   type=float, default=0.01,    help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--tag",        type=str,   default="",      help="Suffix for model filename, e.g. 'v2' → ppo_capacity_v2.zip")
    parser.add_argument("--output_dir", type=str,   default="models",help="Output directory (default: models)")
    args = parser.parse_args()

    with open(log_file, "w") as f:
        f.write("")

    logger.info("=" * 60)
    logger.info("TRAINING RUN STARTED")
    logger.info("=" * 60)

    modes = (
        ["selfish", "capacity", "proximity", "reciprocity", "capacity_proximity"]
        if args.mode == "all" else [args.mode]
    )

    for reward_mode in modes:
        try:
            train_ppo_agent(
                reward_mode=reward_mode,
                output_dir=args.output_dir,
                timesteps=args.timesteps,
                num_envs=args.num_envs,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                tag=args.tag,
            )
        except Exception as e:
            logger.info(f"Error training '{reward_mode}': {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
