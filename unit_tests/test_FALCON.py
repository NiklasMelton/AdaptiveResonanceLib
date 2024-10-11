import pytest
import numpy as np
from artlib.elementary.FuzzyART import FuzzyART
from artlib.reinforcement.FALCON import FALCON


@pytest.fixture
def falcon_model():
    # Initialize FALCON with three FuzzyART modules
    state_art = FuzzyART(0.5, 0.01, 1.0)
    action_art = FuzzyART(0.7, 0.01, 1.0)
    reward_art = FuzzyART(0.9, 0.01, 1.0)
    channel_dims = [4, 4, 2]
    return FALCON(state_art=state_art, action_art=action_art, reward_art=reward_art, channel_dims=channel_dims)


def test_falcon_initialization(falcon_model):
    # Test that the model initializes correctly
    assert isinstance(falcon_model.fusion_art.modules[0], FuzzyART)
    assert isinstance(falcon_model.fusion_art.modules[1], FuzzyART)
    assert isinstance(falcon_model.fusion_art.modules[2], FuzzyART)
    assert falcon_model.fusion_art.channel_dims == [4, 4, 2]


def test_falcon_fit(falcon_model):
    # Test the fit method of FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(states, actions, rewards)

    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    assert len(falcon_model.fusion_art.W) > 0
    assert falcon_model.fusion_art.labels_.shape[0] == states_prep.shape[0]


def test_falcon_partial_fit(falcon_model):
    # Test the partial_fit method of FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(states, actions, rewards)

    falcon_model.partial_fit(states_prep, actions_prep, rewards_prep)

    assert len(falcon_model.fusion_art.W) > 0
    assert falcon_model.fusion_art.labels_.shape[0] == states_prep.shape[0]


def test_falcon_get_actions_and_rewards(falcon_model):
    # Test the get_actions_and_rewards method
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    print("actions\n", actions)
    print("actions_min", np.min(actions,axis=0))
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(states, actions, rewards)
    print("actions_prep\n", actions_prep)

    falcon_model.fit(states_prep, actions_prep, rewards_prep)
    print(states_prep[0,:])

    action_space, rewards = falcon_model.get_actions_and_rewards(states_prep[0,:])

    assert action_space.shape[0] > 0
    assert rewards.shape[0] > 0


# def test_falcon_get_action(falcon_model):
#     # Test the get_action method of FALCON
#     states = np.random.rand(10, 2)
#     actions = np.random.rand(10, 2)
#     rewards = np.random.rand(10, 1)
#
#     # Prepare data
#     states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(states, actions, rewards)
#
#     falcon_model.fit(states_prep, actions_prep, rewards_prep)
#
#     action = falcon_model.get_action(states_prep[0,:])
#
#     assert action.shape[0] == actions.shape[1]

#
# def test_falcon_get_probabilistic_action(falcon_model):
#     # Test the get_probabilistic_action method of FALCON
#     states = np.random.rand(10, 2)
#     actions = np.random.rand(10, 2)
#     rewards = np.random.rand(10, 1)
#
#     # Prepare data
#     states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(states, actions, rewards)
#
#     falcon_model.fit(states_prep, actions_prep, rewards_prep)
#
#     action = falcon_model.get_probabilistic_action(states_prep[0,:])
#
#     assert action.shape[0] == actions.shape[1]


def test_falcon_get_rewards(falcon_model):
    # Test the get_rewards method of FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = falcon_model.prepare_data(states, actions, rewards)

    print(states_prep.shape)
    print(actions_prep.shape)
    print(rewards_prep.shape)

    falcon_model.fit(states_prep, actions_prep, rewards_prep)

    predicted_rewards = falcon_model.get_rewards(states_prep, actions_prep)

    assert predicted_rewards.shape == rewards.shape
