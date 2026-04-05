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
│   ├── plot_results.py           # Visualization script (NEW)
│   ├── analyze_selfish.py        # Selfish agent analysis (NEW)
│   ├── models/                   # Trained PPO agents (generated)
│   ├── results/                  # Evaluation metrics (generated)
│   └── [other model files]
├── figures/                       # Generated publication-quality plots
│   ├── fig_metric*.png           # Comparison figures for all 5 metrics
│   ├── selfish_helping_by_state.png
│   └── selfish_reciprocity_comparison.png
├── analysis/                      # Data cleaning and analysis notebooks
│   └── clean_raw_data.ipynb
├── cogsci24-data-analysis.ipynb  # Analysis notebooks
├── model-fitting.ipynb
├── requirements.txt               # Legacy Python dependencies
├── environment.yml                # Conda environment (updated)
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

Train separate agents for five reward modes (selfish, capacity, proximity, reciprocity, capacity_proximity):

```bash
python train_ppo.py
```

This will:
- Load human sessions from `data/trialdf.csv`
- Train 5 agents with different reward shaping functions
- Save models to `models/ppo_{reward_mode}.zip` 
- Log output to `training_output.txt`
- Takes ~3-6 hours depending on hardware

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

### 4. Generate Comparison Figures

Create publication-quality figures comparing all agents:

```bash
python plot_results.py
```

This will generate 5 figures in `figures/`, each with 6 subplots (5 agent modes + human reference):
- `fig_metric1_backpack.png` - Helping rate by backpack fill
- `fig_metric2_patchuniformity.png` - Helping rate by patch uniformity
- `fig_metric3_distance.png` - Helping rate by distance to partner vegetables
- `fig_metric4_energy.png` - Helping rate by remaining energy
- `fig_metric5_reciprocity.png` - Helping rate by turn (conditional on partner help)

### 5. Analyze Selfish Agent (Extension 2)

Investigate why the selfish agent exhibits reciprocity despite no reciprocity reward:

```bash
python analyze_selfish.py
```

This will:
- Run the selfish agent on all human game replays
- Collect per-turn features (helping, partner helping, backpack, energy, distance, etc.)
- Generate analysis figures in `figures/`:
  - `selfish_helping_by_state.png` - 2×3 panel of helping patterns by state features
  - `selfish_reciprocity_comparison.png` - Reciprocity gap visualization
- Save raw per-turn data to `results/selfish_agent_turns.csv`
- Print summary statistics to stdout

## Reward Modes

The agents are trained with five different reward structures:

| Mode | Description | Hypothesis |
|------|-------------|-----------|
| **Selfish** | No shaping; reward = final score × energy | Baseline: pure self-interest |
| **Capacity** | Bonus when picking partner veggies with spare backpack capacity | Tests if humans help when they have capacity |
| **Proximity** | Bonus inversely proportional to distance to partner vegetables | Tests if proximity drives helping |
| **Reciprocity** | Bonus based on partner's recent helping history | Tests if humans reciprocate help |
| **Capacity+Proximity** | Mixture: 0.5 × capacity_bonus + 0.5 × proximity_bonus | Tests if combined reward better matches human behavior |

## Results

Summary of overall helping rates across all trained agents:

```
Capacity:           7.2% (best match to humans)
Reciprocity:        7.1%
Proximity:          6.9%
Capacity+Proximity: 5.9% (mixture reward)
Selfish:            3.8%
Humans:             8.2% (baseline)
```

**Key Finding:** The **capacity-based reward agent (7.2%)** most closely reproduces human behavior, suggesting **backpack capacity surplus is the primary driver of spontaneous helping**. 

**Extension 1 Result:** The capacity+proximity mixture (5.9%) underperforms compared to capacity alone, indicating that adding proximity penalties reduces behavior alignment with humans. This suggests humans prioritize capacity availability over distance when deciding to help.

**Extension 2 Discovery:** The selfish agent exhibits near-zero reciprocity (-0.2%) despite having no reciprocity reward. Analysis shows:
- Helping rate when partner helped: 6.0%
- Helping rate when partner did not help: 8.0%
- **Actual reciprocal effect is slightly negative**, suggesting the agent helps less when partner helped
- This counterintuitive pattern arises from learned capacity and proximity heuristics, not explicit reciprocal logic

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

- **`plot_results.py`** - Visualization script
  - Loads metric CSVs from evaluation results
  - Generates 5 publication-quality figures (one per metric)
  - Each figure has 6 subplots: 5 agent modes + human reference panel
  - Uses consistent color scheme and formatting for easy comparison
  
- **`analyze_selfish.py`** - Selfish agent analysis (Extension 2)
  - Investigates unexpected reciprocity in the selfish agent
  - Runs selfish agent on all human game replays collecting per-turn features
  - Generates 2×3 panel figure showing helping patterns by state factors
  - Computes reciprocity gap and analyzes state features affecting helping
  - Outputs `selfish_agent_turns.csv` with raw per-turn data for further analysis

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

## Extensions

### Extension 1: Capacity+Proximity Mixture Reward
A fifth agent is trained with a mixture of capacity and proximity rewards (50/50):
```python
reward = 0.5 * capacity_bonus + 0.5 * proximity_bonus
```

**Training & Evaluation:**
- Model: `models/ppo_capacity_proximity.zip`
- Metrics: `results/metrics_capacity_proximity_metric*.csv` (5 behavioral metrics)
- Overall helping rate: **5.9%**

**Finding:** Combining capacity and proximity rewards produces *worse* human alignment (5.9%) compared to capacity alone (7.2%). This suggests that proximity penalties actual *reduce* helping behavior, and humans prioritize backpack capacity surplus over closeness to vegetables when deciding to help.

**Behavioral Pattern Difference:**
- Capacity only: Strong helping when low-fill backpack available
- Cap+Prox hybrid: Penalizes distant vegetables, suppressing helping despite capacity

### Extension 2: Selfish Agent Analysis  
The selfish agent shows unexpected patterns in reciprocity despite having *zero* reciprocity reward shaping. `analyze_selfish.py` investigates by:
- Running selfish agent (`models/ppo_selfish.zip`) on all human game replays
- Collecting per-turn features: backpack fill, energy, distance to partner, history
- Analyzing 2×3 panel of helping patterns by state factors
- Computing reciprocity metrics and their drivers

**Results:**
- Output: `selfish_agent_analysis.txt` (summary statistics)
- Per-turn data: `selfish_agent_turns.csv` (raw features for further analysis)
- Figures: 
  - `selfish_helping_by_state.png` - Helping patterns by BP fill, energy, distance, partner need, turn #, and reciprocity
  - `selfish_reciprocity_comparison.png` - Side-by-side comparison of helping rates

**Key Discovery:**
The selfish agent exhibits **anti-reciprocity**:
- Helping rate when partner helped last: 6.0%
- Helping rate when partner did NOT help: 8.0%
- **Net effect: -2.0%** (agent helps *less* when partner helped)

This counterintuitive pattern shows that emergent social behavior is *not* a simple product of the reward structure, but arises from complex interactions between learned capacity heuristics, state representations, and game dynamics. The agent learns to help based on backpack capacity and item proximity, not social history.

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

- [x] Mixture of reward functions (capacity + proximity) → **Extension 1: Implemented**
- [x] Analyze unexpected reciprocity in selfish agent → **Extension 2: Implemented in `analyze_selfish.py`**
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
