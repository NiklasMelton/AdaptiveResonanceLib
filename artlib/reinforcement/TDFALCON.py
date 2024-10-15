"""
Tan, A.-H., Lu, N., & Xiao, D. (2008). Integrating Temporal Difference Methods and Self-Organizing Neural
Networks for Reinforcement Learning With Delayed Evaluative Feedback. IEEE Transactions on Neural
Networks, 19 , 230–244. doi:10.1109/TNN.2007.905839
"""

import numpy as np
from typing import Optional, List, Union
from artlib.common.BaseART import BaseART
from artlib.common.utils import compliment_code, de_compliment_code
from artlib.reinforcement.FALCON import FALCON

class TD_FALCON(FALCON):
    """TD-FALCON for Reinforcement Learning

    This module implements TD-FALCON as first described in
    Tan, A.-H., Lu, N., & Xiao, D. (2008). Integrating Temporal Difference Methods and Self-Organizing Neural
    Networks for Reinforcement Learning With Delayed Evaluative Feedback. IEEE Transactions on Neural
    Networks, 19 , 230–244. doi:10.1109/TNN.2007.905839.
    TD-FALCON is based on a FALCON backbone but includes specific function for temporal-difference learning.
    Currently, only SARSA is implemented and only Fuzzy ART base modules are supported.


    Parameters:
        state_art: BaseART the instantiated ART module that wil cluster the state-space
        action_art: BaseART the instantiated ART module that wil cluster the action-space
        reward_art: BaseART the instantiated ART module that wil cluster the reward-space
        gamma_values: Union[List[float], np.ndarray] the activation ratio for each channel
        channel_dims: Union[List[int], np.ndarray] the dimension of each channel
        td_alpha: float the learning rate for the temporal difference estimator
        td_lambda: float the future-cost factor

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
        """
        Parameters:
        - state_art: BaseART the instantiated ART module that wil cluster the state-space
        - action_art: BaseART the instantiated ART module that wil cluster the action-space
        - reward_art: BaseART the instantiated ART module that wil cluster the reward-space
        - gamma_values: Union[List[float], np.ndarray] the activation ratio for each channel
        - channel_dims: Union[List[int], np.ndarray] the dimension of each channel
        - td_alpha: float the learning rate for the temporal difference estimator
        - td_lambda: float the future-cost factor
        """
        self.td_alpha = td_alpha
        self.td_lambda = td_lambda
        super(TD_FALCON, self).__init__(state_art, action_art, reward_art, gamma_values, channel_dims)

    def fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        raise NotImplementedError("TD-FALCON can only be trained with partial fit")

    def calculate_SARSA(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, single_sample_reward: Optional[float] = None):
        # calculate SARSA values
        rewards_dcc = de_compliment_code(rewards)
        if len(states) > 1:

            if hasattr(self.fusion_art.modules[0], "W"):
                # if FALCON has been trained get predicted rewards
                Q = self.get_rewards(states, actions)
            else:
                # otherwise set predicted rewards to 0
                Q = np.zeros_like(rewards_dcc)
            # SARSA equation
            sarsa_rewards = Q[:-1] + self.td_alpha * (rewards_dcc[:-1] + self.td_lambda * Q[1:] - Q[:-1])
            # ensure SARSA values are between 0 and 1
            sarsa_rewards = np.maximum(np.minimum(sarsa_rewards, 1.0), 0.0)
            # compliment code rewards
            sarsa_rewards_fit = compliment_code(sarsa_rewards)
            # we cant train on the final state because no rewards are generated after it
            states_fit = states[:-1, :]
            actions_fit = actions[:-1, :]
        else:
            # if we only have a single sample, we cant learn from future samples
            if single_sample_reward is None:
                sarsa_rewards_fit = rewards
            else:
                sarsa_rewards_fit = compliment_code(np.array([[single_sample_reward]]))
            states_fit = states
            actions_fit = actions

        return states_fit, actions_fit, sarsa_rewards_fit

    def partial_fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, single_sample_reward: Optional[float] = None):

        states_fit, actions_fit, sarsa_rewards_fit = self.calculate_SARSA(states, actions, rewards, single_sample_reward)
        data = self.fusion_art.join_channel_data([states_fit, actions_fit, sarsa_rewards_fit])
        self.fusion_art = self.fusion_art.partial_fit(data)
        return self

