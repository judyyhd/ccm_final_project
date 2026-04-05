import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import farmgame
import utils


class FarmEnv(gym.Env):
    """
    Gymnasium environment wrapping the farming game for PPO training.
    The agent plays as the specified color, while the partner's actions
    come from a pre-loaded human replay.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        human_game: list,
        agent_color: str = "red",
        reward_mode: str = "selfish",
        history_window: int = 3,
    ):
        """
        Args:
            human_game: List of Transition objects from a human replay (one game).
            agent_color: "red" or "purple" (which side the RL agent plays).
            reward_mode: "selfish", "capacity", "proximity", or "reciprocity".
            history_window: Number of past turns to include in history vector.
        """
        self.human_game = human_game
        self.agent_color = agent_color
        self.reward_mode = reward_mode
        self.history_window = history_window

        # Grid dimensions
        self.grid_width = 20
        self.grid_height = 20

        # Maximum number of items on farm (upper bound)
        self.max_items = 14

        # Action space: Discrete with up to 16 possible actions
        self.action_space = spaces.Discrete(16)

        # Observation space dimensions:
        # 1. Agent position (2)
        # 2. Partner position (2)
        # 3. Agent energy (1)
        # 4. Agent backpack fill (1)
        # 5. Partner backpack fill (1)
        # 6. Items (max_items * 4): [x_norm, y_norm, is_own_color, is_farm]
        # 7. History window (history_window)
        obs_size = (
            2  # agent pos
            + 2  # partner pos
            + 1  # agent energy
            + 1  # agent backpack fill
            + 1  # partner backpack fill
            + (self.max_items * 4)  # items
            + self.history_window  # history
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Track state and replay
        self.state = None
        self.replay_index = 0  # Index into human_game
        self.partner_helped_history = [
            False for _ in range(history_window)
        ]  # Rolling window

    def reset(self, seed=None, options=None):
        """Reset environment to initial state of a new game."""
        super().reset(seed=seed)

        # Reset replay index
        self.replay_index = 0
        self.partner_helped_history = [False for _ in range(self.history_window)]

        # Sample a new game config using the first state from human replay
        if len(self.human_game) > 0:
            initial_state = self.human_game[0].state
            self.state = copy.deepcopy(initial_state)
        else:
            # Fallback: create a default game
            self.state = farmgame.configure_game()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action_idx: int):
        """
        Execute one step of the environment.

        Args:
            action_idx: Integer action index (0 to action_space.n - 1).

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Ensure we only act on the agent's turn
        if self.state.whose_turn()["color"] != self.agent_color:
            raise RuntimeError(
                f"It is not {self.agent_color}'s turn. Current turn: {self.state.whose_turn()['color']}"
            )

        # Map action index to legal action
        legal_actions = self.state.legal_actions()
        if action_idx >= len(legal_actions):
            action_idx = len(legal_actions) - 1  # Use last legal action as fallback
        agent_action = legal_actions[action_idx]

        # Take agent's action
        self.state = self.state.take_action(agent_action, inplace=False)

        # Check if game is done
        if self.state.is_done():
            reward = self._compute_reward(agent_action, is_final=True)
            obs = self._get_obs()
            return obs, reward, True, False, {}

        # Auto-step through partner's turn
        partner_color = "purple" if self.agent_color == "red" else "red"
        while self.state.whose_turn()["color"] == partner_color and not self.state.is_done():
            # Get partner's action from replay
            partner_action = self._get_partner_action()
            partner_helped_this_action = (
                farmgame.Transition(self.state, partner_action).is_helping(partner_color)
            )
            # Track if partner helped on this turn
            self.partner_helped_history.insert(0, partner_helped_this_action)
            self.partner_helped_history = self.partner_helped_history[:self.history_window]

            self.state = self.state.take_action(partner_action, inplace=False)

        # Compute reward for agent's action
        reward = self._compute_reward(agent_action, is_final=self.state.is_done())

        # Get new observation
        obs = self._get_obs()

        # Check if game is over after partner's turns
        terminated = self.state.is_done()

        return obs, reward, terminated, False, {}

    def _get_partner_action(self) -> farmgame.Action:
        """
        Get the next partner action from the human replay.
        If replay is exhausted, use first legal action as fallback.
        """
        if self.replay_index < len(self.human_game):
            transition = self.human_game[self.replay_index]
            self.replay_index += 1
            # Return the action from the replay
            return transition.action
        else:
            # Fallback: use first legal action
            legal_actions = self.state.legal_actions()
            return legal_actions[0] if legal_actions else None

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current state."""
        obs = []

        agent = self.state.playersDict[self.agent_color]
        partner = self.state.playersDict[
            "purple" if self.agent_color == "red" else "red"
        ]

        # Agent position (normalized)
        obs.append(agent["loc"]["x"] / self.grid_width)
        obs.append(agent["loc"]["y"] / self.grid_height)

        # Partner position (normalized)
        obs.append(partner["loc"]["x"] / self.grid_width)
        obs.append(partner["loc"]["y"] / self.grid_height)

        # Agent energy (normalized)
        obs.append(agent["energy"] / 100.0)

        # Agent backpack fill
        obs.append(len(agent["backpack"]["contents"]) / agent["backpack"]["capacity"])

        # Partner backpack fill
        obs.append(len(partner["backpack"]["contents"]) / partner["backpack"]["capacity"])

        # Items (up to max_items)
        items_padded = self.state.items + [None] * (
            self.max_items - len(self.state.items)
        )
        for i in range(self.max_items):
            item = items_padded[i]
            if item is not None:
                obs.append(item.loc["x"] / self.grid_width)
                obs.append(item.loc["y"] / self.grid_height)
                is_own_color = 1.0 if item.color == self.agent_color else 0.0
                obs.append(is_own_color)
                is_farm = 1.0 if item.status == "farm" else 0.0
                obs.append(is_farm)
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        # History window
        for helped in self.partner_helped_history:
            obs.append(1.0 if helped else 0.0)

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, action: farmgame.Action, is_final: bool) -> float:
        """
        Compute the shaped reward based on reward_mode.

        Args:
            action: The action just taken by the agent.
            is_final: Whether the game just ended.

        Returns:
            Reward scalar.
        """
        reward = 0.0

        if self.reward_mode == "selfish":
            if is_final:
                agent = self.state.playersDict[self.agent_color]
                reward = agent["bonuspoints"]

        elif self.reward_mode == "capacity":
            reward = self._compute_capacity_reward(action, is_final)

        elif self.reward_mode == "proximity":
            reward = self._compute_proximity_reward(action, is_final)

        elif self.reward_mode == "reciprocity":
            reward = self._compute_reciprocity_reward(action, is_final)

        return reward

    def _compute_capacity_reward(self, action: farmgame.Action, is_final: bool) -> float:
        """Reward = selfish + capacity bonus when picking partner veggies with spare capacity."""
        reward = 0.0

        if is_final:
            agent = self.state.playersDict[self.agent_color]
            reward = agent["bonuspoints"]
        else:
            # Check if action is a helping action (picking partner's veggie)
            if (
                action.type == farmgame.ActionType.veggie
                and action.color != self.agent_color
            ):
                # Check capacity at the time of action (before adding to backpack)
                # We need to look at the state before this action was taken
                agent = self.state.playersDict[self.agent_color]
                capacity = agent["backpack"]["capacity"]
                # Since action already added the item, backpack has len+1
                num_items_before = len(agent["backpack"]["contents"]) - 1
                spare_capacity = capacity - num_items_before
                reward = spare_capacity / capacity

        return reward

    def _compute_proximity_reward(self, action: farmgame.Action, is_final: bool) -> float:
        """Reward = selfish + proximity bonus when picking partner veggies."""
        reward = 0.0

        if is_final:
            agent = self.state.playersDict[self.agent_color]
            reward = agent["bonuspoints"]
        else:
            # Check if action is a helping action
            if (
                action.type == farmgame.ActionType.veggie
                and action.color != self.agent_color
            ):
                agent = self.state.playersDict[self.agent_color]
                # Find nearest partner vegetable on farm
                partner_color = "purple" if self.agent_color == "red" else "red"
                partner_veggies = [
                    item
                    for item in self.state.items
                    if item.color == partner_color and item.status == "farm"
                ]
                if partner_veggies:
                    min_dist = min(
                        utils.getManhattanDistance(agent["loc"], veg.loc)
                        for veg in partner_veggies
                    )
                    reward = 1.0 / (1.0 + min_dist)

        return reward

    def _compute_reciprocity_reward(self, action: farmgame.Action, is_final: bool) -> float:
        """Reward = selfish + reciprocity bonus when picking partner veggies."""
        reward = 0.0

        if is_final:
            agent = self.state.playersDict[self.agent_color]
            reward = agent["bonuspoints"]
        else:
            # Check if action is a helping action
            if (
                action.type == farmgame.ActionType.veggie
                and action.color != self.agent_color
            ):
                # Fraction of recent turns where partner helped
                if self.history_window > 0:
                    num_helped = sum(self.partner_helped_history)
                    reward = num_helped / self.history_window

        return reward
