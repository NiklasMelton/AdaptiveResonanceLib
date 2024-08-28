import numpy as np
from typing import Optional, Literal
from artlib import FusionART, BaseART


class FALCON:
    def __init__(
            self,
            state_art: BaseART,
            action_art: BaseART,
            reward_art: BaseART,
            gamma_values: list[float] = np.array([0.33, 0.33, 0.34]),
            channel_dims = list[int]
    ):
        self.fusion_art = FusionART(
            modules=[state_art, action_art, reward_art],
            gamma_values=gamma_values,
            channel_dims=channel_dims
        )

    def fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        data = self.fusion_art.join_channel_data([states, actions, rewards])
        self.fusion_art = self.fusion_art.fit(data)
        return self

    def partial_fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        data = self.fusion_art.join_channel_data([states, actions, rewards])
        self.fusion_art = self.fusion_art.partial_fit(data)
        return self

    def get_actions_and_rewards(self, state: np.ndarray, action_space: Optional[np.ndarray] = None) -> np.ndarray:
        reward_centers = self.fusion_art.get_channel_centers(2)
        if action_space is None:
            action_space = self.fusion_art.get_channel_centers(1)
            action_space = np.array(action_space)
        action_space_prepared = self.fusion_art.modules[0].prepare_data(action_space)
        viable_clusters = []
        for action in action_space_prepared:
            data = self.fusion_art.join_channel_data([state.reshape(1, -1), action.reshape(1, -1)], skip_channels=[2])
            c = self.fusion_art.predict(data, skip_channels=[2])
            viable_clusters.append(c[0])

        rewards = [reward_centers[c] for c in viable_clusters]

        return action_space, np.array(rewards)
    def get_action(self, state: np.ndarray, action_space: Optional[np.ndarray] = None, optimality: Literal["min", "max"] = "max") -> np.ndarray:
        action_space, rewards = self.get_actions_and_rewards(state, action_space)
        if optimality == "max":
            c_winner = np.argmax(rewards)
        else:
            c_winner = np.argmin(rewards)
        return action_space[c_winner]

    def get_probabilistic_action(self, state: np.ndarray, action_space: Optional[np.ndarray] = None, offset: float = 0.1, optimality: Literal["min", "max"] = "max") -> np.ndarray:
        action_space, rewards = self.get_actions_and_rewards(state, action_space)
        action_indices = np.array(range(len(action_space)))


        reward_dist = rewards
        reward_dist /= np.sum(reward_dist)
        reward_dist = reward_dist.reshape((-1,))

        if optimality == "min":
            reward_dist = 1.-reward_dist

        reward_dist = np.maximum(np.minimum(reward_dist, offset), 0.0001)
        reward_dist /= np.sum(reward_dist)

        a_i = np.random.choice(action_indices, size=1, p=reward_dist)
        return action_space[a_i[0]][0]

    def get_rewards(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        reward_centers = self.fusion_art.get_channel_centers(2)
        data = self.fusion_art.join_channel_data([states, actions], skip_channels=[2])
        C = self.fusion_art.predict(data, skip_channels=[2])
        return np.array([reward_centers[c] for c in C])








