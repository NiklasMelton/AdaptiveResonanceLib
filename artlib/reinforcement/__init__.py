"""Reinforcement learning (RL) is a type of machine learning where agents learn to make
decisions by interacting with an environment and receiving feedback in the form of
rewards or penalties. The SARSA (State-Action-Reward-State-Action) algorithm is an on-
policy RL method that updates the agent’s policy based on the action actually taken.
This contrasts with Q-learning, which is off-policy and learns the optimal action
independently of the agent’s current policy.

Reactive learning, on the other hand, is a more straightforward approach where decisions
are made solely based on immediate observations, without the complex state-action-reward
feedback loop typical of RL models like SARSA or Q-learning. It lacks the depth of
planning and long-term reward optimization seen in traditional RL.

The modules herein only provide for reactive and SARSA style learning.

`SARSA <https://en.wikipedia.org/wiki/Reinforcement_learning#SARSA>`_

`Reactive agents <https://en.wikipedia.org/wiki/Reactive_planning>`_

"""
