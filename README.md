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
  - `state.reward(color)` — returns final score (score × remaining energy) when game is done
  - `state.legal_actions()` — returns list of valid actions for current player
  - `state.take_action(action)` — updates state and returns new state
- **`modeling/farmgame_io.py`** - Load human sessions from CSV

### New RL Pipeline (This Project)
- **`farm_env.py`** - Gymnasium environment wrapper
  - **Observation space** (60-dim float32): agent position, partner position, energy, backpack fill, items (location, color ownership, status), partner helping history
  - **Action space**: Discrete(16) mapped to legal actions; invalid actions fallback to last legal action
  - **Grid size**: 26×26 (normalized to [0,1])
  - **Reward modes**: Selfish, capacity, proximity, reciprocity (see table above)
  - **Key implementation**: Auto-steps through partner turns using human replay; captures pre-action backpack state for accurate capacity rewards
  
- **`train_ppo.py`** - Training script
  - Loads all human sessions and creates vectorized environments
  - Uses `stable-baselines3.PPO` with MlpPolicy
  - Hyperparameters: `n_steps=512, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01`
  - 500k timesteps per reward mode (~2-4 hours per mode on CPU)
  - Saves trained models to `models/ppo_{reward_mode}.zip`
  
- **`evaluate.py`** - Evaluation script
  - Runs trained agents through human replay data and collects behavioral metrics
  - Filters human data to match agent color and computes 5 metrics
  - Outputs comparison CSVs and summary statistics

## Implementation Details

### Reward Computation

All reward modes follow this pattern:
- **At final state**: `reward = state.reward(agent_color)` (score × energy)
- **At intermediate steps**: Mode-specific shaping of helping actions (picking partner vegetables)

**Capacity reward** uses pre-action backpack state:
```
if helping_action:
    spare_capacity = (capacity - pre_action_contents) / capacity
    reward = spare_capacity
```

This ensures fair measurement of spare capacity at decision time, not after the item is picked up.

### Environment Gotchas

1. **Grid coordinates**: Game uses 0–25 range; normalized to [0,1] for neural network
2. **Turn handling**: Reset() auto-steps through any partner turns at game start
3. **Replay exhaustion**: If human replay runs out, environment falls back to first legal action
4. **Helping definition**: Must pick up *partner-colored* vegetable from *farm* status (not already in backpack)

## Bug Fixes (Version 2)

Recent fixes applied to improve correctness:
- ✅ Replaced nonexistent `playersDict` with direct `state.redplayer`/`state.purpleplayer` access
- ✅ Implemented proper reward mode dispatcher instead of single computation path
- ✅ Use `state.reward()` instead of unreliable `bonuspoints` field
- ✅ Moved partner turn auto-stepping to `reset()` to prevent double-advancing replay index
- ✅ Increased grid dimensions to 26 to handle full coordinate range
- ✅ Capture pre-action backpack state for accurate capacity reward calculation

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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: data/trialdf.csv` | Run from `modeling/` directory; use relative path `../data/trialdf.csv` |
| `AttributeError: 'dict' object has no attribute 'predict'` | Update to latest farm_env.py; variable shadowing bug fixed |
| `KeyError: 'ownBPsize'` | CSV columns may vary; evaluate.py uses `.get()` with fallback defaults |
| Models not saving | Ensure `models/` directory exists; train_ppo.py creates it automatically |
| Grid normalization errors | Grid is 26×26; if you see coordinates >25, increase GRID_WIDTH/GRID_HEIGHT |

## References

Popp, O., & Gureckis, M. (2024). Spontaneous cooperation in turn-based games. *CogSci 2024 Proceedings*.

## Contact

Huizhen Jin, Shengduo Li, Judy Yang - DS-GA 1016 Final Project, Spring 2026
