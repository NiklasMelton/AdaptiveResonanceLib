"""FALCON :cite:`tan2004falcon`, :cite:`tan2008integrating`."""
# Tan, A.-H. (2004).
# FALCON: a fusion architecture for learning, cognition, and navigation.
# In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
# (pp. 3297–3302). volume 4. doi:10.1109/ IJCNN.2004.1381208.

# Tan, A.-H., Lu, N., & Xiao, D. (2008).
# Integrating Temporal Difference Methods and Self-Organizing Neural Networks for
# Reinforcement Learning With Delayed Evaluative Feedback.
# IEEE Transactions on Neural Networks, 19 , 230–244. doi:10.1109/TNN.2007.905839

import numpy as np
from typing import Optional, Literal, Tuple, Union, List
from artlib.common.BaseART import BaseART
from artlib.common.utils import complement_code, de_complement_code
from artlib.fusion.FusionART import FusionART


class FALCON:
    """FALCON for Reinforcement Learning.

    This module implements the reactive FALCON as first described in:
    :cite:`tan2004falcon`.

    .. # Tan, A.-H. (2004).
    .. # FALCON: a fusion architecture for learning, cognition, and navigation.
    .. # In Proc. IEEE International Joint Conference on Neural Networks (IJCNN)
    .. # (pp. 3297–3302). volume 4. doi:10.1109/ IJCNN.2004.1381208.

    FALCON is based on a :class:`~artlib.fusion.FusionART.FusionART` backbone but only
    accepts 3 channels: State, Action, and Reward. Specific functions are implemented
    for getting optimal reward and action predictions.

    """

    def __init__(
        self,
        state_art: BaseART,
        action_art: BaseART,
        reward_art: BaseART,
        gamma_values: Union[List[float], np.ndarray] = np.array([0.33, 0.33, 0.34]),
        channel_dims: Union[List[int], np.ndarray] = list[int],
    ):
        """Initialize the FALCON model.

        Parameters
        ----------
        state_art : BaseART
            The instantiated ART module that will cluster the state-space.
        action_art : BaseART
            The instantiated ART module that will cluster the action-space.
        reward_art : BaseART
            The instantiated ART module that will cluster the reward-space.
        gamma_values : list of float or np.ndarray, optional
            The activation ratio for each channel, by default [0.33, 0.33, 0.34].
        channel_dims : list of int or np.ndarray
            The dimension of each channel.

        """
        self.fusion_art = FusionART(
            modules=[state_art, action_art, reward_art],
            gamma_values=gamma_values,
            channel_dims=channel_dims,
        )

    def prepare_data(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for clustering.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.

        Returns
        -------
        tuple of np.ndarray
            Normalized state, action, and reward data.

        """
        return (
            self.fusion_art.modules[0].prepare_data(states),
            self.fusion_art.modules[1].prepare_data(actions),
            self.fusion_art.modules[2].prepare_data(rewards),
        )

    def restore_data(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Restore data to its original form before preparation.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.

        Returns
        -------
        tuple of np.ndarray
            Restored state, action, and reward data.

        """
        return (
            self.fusion_art.modules[0].restore_data(states),
            self.fusion_art.modules[1].restore_data(actions),
            self.fusion_art.modules[2].restore_data(rewards),
        )

    def fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Fit the FALCON model to the data.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.

        Returns
        -------
        FALCON
            The fitted FALCON model.

        """
        data = self.fusion_art.join_channel_data([states, actions, rewards])
        self.fusion_art = self.fusion_art.fit(data)
        return self

    def partial_fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Partially fit the FALCON model to the data.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.

        Returns
        -------
        FALCON
            The partially fitted FALCON model.

        """
        data = self.fusion_art.join_channel_data([states, actions, rewards])
        self.fusion_art = self.fusion_art.partial_fit(data)
        return self

    def get_actions_and_rewards(
        self, state: np.ndarray, action_space: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get possible actions and their associated rewards for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state.
        action_space : np.ndarray, optional
            The available action space, by default None.

        Returns
        -------
        tuple of np.ndarray
            The possible actions and their corresponding rewards.

        """
        reward_centers = self.fusion_art.get_channel_centers(2)
        if action_space is None:
            action_space = self.fusion_art.get_channel_centers(1)
            action_space = np.array(action_space)
        action_space_prepared = self.fusion_art.modules[1].prepare_data(action_space)
        viable_clusters = []
        for action in action_space_prepared:
            data = self.fusion_art.join_channel_data(
                [state.reshape(1, -1), action.reshape(1, -1)], skip_channels=[2]
            )
            c = self.fusion_art.predict(data, skip_channels=[2])
            viable_clusters.append(c[0])

        rewards = [reward_centers[c] for c in viable_clusters]

        return action_space, np.array(rewards)

    def get_action(
        self,
        state: np.ndarray,
        action_space: Optional[np.ndarray] = None,
        optimality: Literal["min", "max"] = "max",
    ) -> np.ndarray:
        """Get the best action for a given state based on optimality.

        Parameters
        ----------
        state : np.ndarray
            The current state.
        action_space : np.ndarray, optional
            The available action space, by default None.
        optimality : {"min", "max"}, optional
            Whether to choose the action with the minimum or maximum reward,
            by default "max".

        Returns
        -------
        np.ndarray
            The optimal action.

        """
        action_space, rewards = self.get_actions_and_rewards(state, action_space)
        if optimality == "max":
            c_winner = np.argmax(rewards)
        else:
            c_winner = np.argmin(rewards)
        return action_space[c_winner]

    def get_probabilistic_action(
        self,
        state: np.ndarray,
        action_space: Optional[np.ndarray] = None,
        offset: float = 0.1,
        optimality: Literal["min", "max"] = "max",
    ) -> np.ndarray:
        """Get a probabilistic action for a given state based on reward distribution.

        Parameters
        ----------
        state : np.ndarray
            The current state.
        action_space : np.ndarray, optional
            The available action space, by default None.
        offset : float, optional
            The reward offset to adjust probability distribution, by default 0.1.
        optimality : {"min", "max"}, optional
            Whether to prefer minimum or maximum rewards, by default "max".

        Returns
        -------
        np.ndarray
            The chosen action based on probability.

        """
        action_space, rewards = self.get_actions_and_rewards(state, action_space)
        action_indices = np.array(range(len(action_space)))

        reward_dist = rewards
        reward_dist /= np.sum(reward_dist)
        reward_dist = reward_dist.reshape((-1,))

        if optimality == "min":
            reward_dist = 1.0 - reward_dist

        reward_dist = np.maximum(np.minimum(reward_dist, offset), 0.0001)
        reward_dist /= np.sum(reward_dist)

        a_i = np.random.choice(action_indices, size=1, p=reward_dist)
        return action_space[a_i[0]][0]

    def get_rewards(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Get the rewards for given states and actions.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.

        Returns
        -------
        np.ndarray
            The rewards corresponding to the given state-action pairs.

        """
        reward_centers = self.fusion_art.get_channel_centers(2)
        data = self.fusion_art.join_channel_data([states, actions], skip_channels=[2])
        C = self.fusion_art.predict(data, skip_channels=[2])
        return np.array([reward_centers[c] for c in C])


class TD_FALCON(FALCON):
    """TD-FALCON for Reinforcement Learning.

    This module implements TD-FALCON as first described in:
    :cite:`tan2008integrating`.

    .. # Tan, A.-H., Lu, N., & Xiao, D. (2008).
    .. # Integrating Temporal Difference Methods and Self-Organizing Neural Networks for
    .. # Reinforcement Learning With Delayed Evaluative Feedback.
    .. # IEEE Transactions on Neural Networks, 19 , 230–244. doi:10.1109/TNN.2007.905839

    TD-FALCON is based on a :class:`FALCON` backbone but includes specific function for
    temporal-difference learning. Currently, only SARSA is implemented and only
    :class:`~artlib.elementary.FuzzyART.FuzzyART` base modules are supported.

    """

    def __init__(
        self,
        state_art: BaseART,
        action_art: BaseART,
        reward_art: BaseART,
        gamma_values: Union[List[float], np.ndarray] = np.array([0.33, 0.33, 0.34]),
        channel_dims: Union[List[int], np.ndarray] = list[int],
        td_alpha: float = 1.0,
        td_lambda: float = 1.0,
    ):
        """Initialize the TD-FALCON model.

        Parameters
        ----------
        state_art : BaseART
            The instantiated ART module that will cluster the state-space.
        action_art : BaseART
            The instantiated ART module that will cluster the action-space.
        reward_art : BaseART
            The instantiated ART module that will cluster the reward-space.
        gamma_values : list of float or np.ndarray, optional
            The activation ratio for each channel, by default [0.33, 0.33, 0.34].
        channel_dims : list of int or np.ndarray
            The dimension of each channel.
        td_alpha : float, optional
            The learning rate for the temporal difference estimator, by default 1.0.
        td_lambda : float, optional
            The future-cost factor for temporal difference learning, by default 1.0.

        """
        self.td_alpha = td_alpha
        self.td_lambda = td_lambda
        super(TD_FALCON, self).__init__(
            state_art, action_art, reward_art, gamma_values, channel_dims
        )

    def fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Fit the TD-FALCON model to the data.

        Raises
        ------
        NotImplementedError
            TD-FALCON can only be trained with partial fit.

        """
        raise NotImplementedError("TD-FALCON can only be trained with partial fit")

    def calculate_SARSA(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        single_sample_reward: Optional[float] = None,
    ):
        """Calculate the SARSA values for reinforcement learning.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.
        single_sample_reward : float, optional
            The reward for a single sample, if applicable, by default None.

        Returns
        -------
        tuple of np.ndarray
            The state, action, and SARSA-adjusted reward data to be used for fitting.

        """
        # calculate SARSA values
        rewards_dcc = de_complement_code(rewards)
        if len(states) > 1:
            if hasattr(self.fusion_art.modules[0], "W"):
                # if FALCON has been trained get predicted rewards
                Q = self.get_rewards(states, actions)
            else:
                # otherwise set predicted rewards to 0
                Q = np.zeros_like(rewards_dcc)
            # SARSA equation
            sarsa_rewards = Q[:-1] + self.td_alpha * (
                rewards_dcc[:-1] + self.td_lambda * Q[1:] - Q[:-1]
            )
            # ensure SARSA values are between 0 and 1
            sarsa_rewards = np.maximum(np.minimum(sarsa_rewards, 1.0), 0.0)
            # complement code rewards
            sarsa_rewards_fit = complement_code(sarsa_rewards)
            # we cant train on the final state because no rewards are generated after it
            states_fit = states[:-1, :]
            actions_fit = actions[:-1, :]
        else:
            # if we only have a single sample, we cant learn from future samples
            if single_sample_reward is None:
                sarsa_rewards_fit = rewards
            else:
                sarsa_rewards_fit = complement_code(np.array([[single_sample_reward]]))
            states_fit = states
            actions_fit = actions

        return states_fit, actions_fit, sarsa_rewards_fit

    def partial_fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        single_sample_reward: Optional[float] = None,
    ):
        """Partially fit the TD-FALCON model using SARSA.

        Parameters
        ----------
        states : np.ndarray
            The state data.
        actions : np.ndarray
            The action data.
        rewards : np.ndarray
            The reward data.
        single_sample_reward : float, optional
            The reward for a single sample, if applicable, by default None.

        Returns
        -------
        TD_FALCON
            The partially fitted TD-FALCON model.

        """
        states_fit, actions_fit, sarsa_rewards_fit = self.calculate_SARSA(
            states, actions, rewards, single_sample_reward
        )
        data = self.fusion_art.join_channel_data(
            [states_fit, actions_fit, sarsa_rewards_fit]
        )
        self.fusion_art = self.fusion_art.partial_fit(data)
        return self
