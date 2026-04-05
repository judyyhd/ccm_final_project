# Computational Cognitive Modeling: Spontaneous Helping in a Farming Game

This project extends [Osborn Popp & Gureckis (2024)](https://arxiv.org/abs/2407.09747), a paper studying spontaneous helping behavior in a two-player farming game. We train PPO reinforcement learning agents to replicate human helping decisions and compare behavioral patterns across different reward assumptions.

## Overview

In this turn-based grid game:
- Two farmers (red and purple) collect colored vegetables
- Players earn points only for their own color (score × remaining energy)
- Players can choose to pick up their partner's vegetables at personal cost
- The project investigates: **Which reward structures produce human-like helping?**

## Project Structure

```
.
├── data/                          # Human participant data
│   ├── trialdf.csv               # Main trial-level data with all behavioral metrics
│   ├── browser_data.csv, captcha_data.csv, ...
│   └── raw/
├── modeling/                      # Game implementation and RL pipeline
│   ├── farmgame.py               # Core game engine
│   ├── farmgame_io.py            # Data loading utilities
│   ├── farm_env.py               # Gymnasium environment wrapper (NEW)
│   ├── train_ppo.py              # PPO training script (NEW)
│   ├── evaluate.py               # Behavioral evaluation script (NEW)
│   ├── models/                   # Trained PPO agents (generated)
│   ├── results/                  # Evaluation metrics (generated)
│   └── [other model files]
├── analysis/                      # Data cleaning and analysis notebooks
│   └── clean_raw_data.ipynb
├── cogsci24-data-analysis.ipynb  # Analysis notebooks
├── model-fitting.ipynb
├── requirements.txt               # Legacy Python dependencies
├── environment.yml                # Conda environment (NEW)
└── README.md
```

## Getting Started

### 1. Set Up Environment

```bash
conda env create -f environment.yml
conda activate ccm-farming-game
cd modeling
```

### 2. Train PPO Agents

Train separate agents for four reward modes (selfish, capacity, proximity, reciprocity):

```bash
python train_ppo.py
```

This will:
- Load human sessions from `data/trialdf.csv`
- Train 4 agents with different reward shaping functions
- Save models to `models/ppo_{reward_mode}.zip` 
- Log output to `training_output.txt`
- Takes ~2-4 hours depending on hardware

### 3. Evaluate Agent Behavior

Compare trained agents against human behavioral data:

```bash
python evaluate.py
```

This will:
- Load trained models and human replay data
- Run agents through all games and collect behavioral metrics
- Generate CSV files in `results/` comparing agent vs. human patterns
- Print summary helping rates for each reward mode
- Log output to `evaluation_output.txt`

## Reward Modes

The agents are trained with four different reward structures:

| Mode | Description | Hypothesis |
|------|-------------|-----------|
| **Selfish** | No shaping; reward = final score × energy | Baseline: pure self-interest |
| **Capacity** | Bonus when picking partner veggies with spare backpack capacity | Tests if humans help when they have capacity |
| **Proximity** | Bonus inversely proportional to distance to partner vegetables | Tests if proximity drives helping |
| **Reciprocity** | Bonus based on partner's recent helping history | Tests if humans reciprocate help |

## Results

Summary of overall helping rates:

```
Capacity:    6.6% (best match to humans)
Reciprocity: 4.6%
Selfish:     3.6%
Proximity:   3.4%
Humans:      7.6% (baseline)
```

**Finding:** The capacity-based reward agent (6.6%) most closely reproduces human behavior, suggesting **backpack capacity surplus is the primary driver of spontaneous helping**.

## Evaluation Metrics

Five behavioral metrics are computed for each agent (in `results/metrics_{reward_mode}_*.csv`):

1. **Helpfulness by backpack size** - Helping rate by capacity (3, 4, 5)
2. **Helpfulness by distance** - Helping rate binned by distance to partner veggies
3. **Helpfulness by remaining energy** - Helping rate by energy deciles
4. **Conditional helping** - Whether agent helps given partner's previous action
5. **Patch uniformity** - Helping rate by veggie distribution uniformity

Each CSV includes agent and human data for direct comparison and plotting.

## Key Files

### Core Game Implementation
- **`modeling/farmgame.py`** - Game state, actions, and rules
- **`modeling/farmgame_io.py`** - Load human sessions from CSV

### New RL Pipeline (This Project)
- **`farm_env.py`** - Gymnasium environment wrapper
  - Observation: agent/partner positions, energy, backpack fill, items, history
  - Action space: Discrete(16) mapped to legal actions
  - Four reward modes for different hypotheses about human motivation
  
- **`train_ppo.py`** - Training script
  - Uses `stable-baselines3.PPO` with MlpPolicy
  - Vectorized environments with human replay data
  - 500k timesteps per reward mode
  
- **`evaluate.py`** - Evaluation script
  - Runs trained agents against human replay data
  - Computes 5 behavioral metrics
  - Outputs comparison CSVs and summary statistics

## Dependencies

Key packages (see `environment.yml`):
- `gymnasium>=0.29.0` - Environment API
- `stable-baselines3>=2.2.0` - PPO and other RL algorithms
- `pandas`, `numpy`, `matplotlib` - Data analysis
- `jupyter` - Notebooks

## Future Work

- [ ] Mixture of reward functions (capacity + reciprocity)
- [ ] Hyperparameter tuning for better human alignment
- [ ] Analyze learned value functions to understand decision-making
- [ ] Test agents in novel environments not seen in training
- [ ] Compare to other RL algorithms (A2C, DQN, etc.)

## References

Popp, O., & Gureckis, M. (2024). Spontaneous cooperation in turn-based games. *CogSci 2024 Proceedings*.

## Contact

Huizhen Jin, Shengduo Li, Judy Yang - DS-GA 1016 Final Project, Spring 2026
