import pytest
import numpy as np
from artlib.elementary.FuzzyART import FuzzyART
from artlib.reinforcement.FALCON import TD_FALCON


@pytest.fixture
def td_falcon_model():
    # Initialize TD_FALCON with three FuzzyART modules
    state_art = FuzzyART(0.5, 0.01, 1.0)
    action_art = FuzzyART(0.7, 0.01, 1.0)
    reward_art = FuzzyART(0.9, 0.01, 1.0)
    channel_dims = [4, 4, 2]
    return TD_FALCON(state_art=state_art, action_art=action_art, reward_art=reward_art, channel_dims=channel_dims)


def test_td_falcon_initialization(td_falcon_model):
    # Test that the TD_FALCON model initializes correctly
    assert isinstance(td_falcon_model.fusion_art.modules[0], FuzzyART)
    assert isinstance(td_falcon_model.fusion_art.modules[1], FuzzyART)
    assert isinstance(td_falcon_model.fusion_art.modules[2], FuzzyART)
    assert td_falcon_model.fusion_art.channel_dims == [4, 4, 2]
    assert td_falcon_model.td_alpha == 1.0
    assert td_falcon_model.td_lambda == 1.0


def test_td_falcon_fit_raises(td_falcon_model):
    # Test that calling the fit method raises NotImplementedError
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    with pytest.raises(NotImplementedError):
        td_falcon_model.fit(states, actions, rewards)


def test_td_falcon_partial_fit(td_falcon_model):
    # Test the partial_fit method of TD_FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = td_falcon_model.prepare_data(states, actions, rewards)

    td_falcon_model.partial_fit(states_prep, actions_prep, rewards_prep)

    assert len(td_falcon_model.fusion_art.W) > 0
    assert td_falcon_model.fusion_art.labels_.shape[0] == states_prep.shape[0]-1


def test_td_falcon_calculate_SARSA(td_falcon_model):
    # Test the calculate_SARSA method of TD_FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    states_prep, actions_prep, rewards_prep = td_falcon_model.prepare_data(states, actions, rewards)

    # Test with multiple samples
    states_fit, actions_fit, sarsa_rewards_fit = td_falcon_model.calculate_SARSA(states_prep, actions_prep, rewards_prep)

    assert states_fit.shape == (9, 4)  # Last sample is discarded for SARSA
    assert actions_fit.shape == (9, 4)
    assert sarsa_rewards_fit.shape == (9, 2)

    # Test with single sample
    states_fit, actions_fit, sarsa_rewards_fit = td_falcon_model.calculate_SARSA(states_prep[:1], actions_prep[:1], rewards_prep[:1])

    assert states_fit.shape == (1, 4)
    assert actions_fit.shape == (1, 4)
    assert sarsa_rewards_fit.shape == (1, 2)


def test_td_falcon_get_actions_and_rewards(td_falcon_model):
    # Test the get_actions_and_rewards method of TD_FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = td_falcon_model.prepare_data(states, actions, rewards)

    td_falcon_model.partial_fit(states_prep, actions_prep, rewards_prep)

    action_space, rewards = td_falcon_model.get_actions_and_rewards(states_prep[0, :])

    assert action_space.shape[0] > 0
    assert rewards.shape[0] > 0


def test_td_falcon_get_action(td_falcon_model):
    # Test the get_action method of TD_FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = td_falcon_model.prepare_data(states, actions, rewards)

    td_falcon_model.partial_fit(states_prep, actions_prep, rewards_prep)

    action = td_falcon_model.get_action(states_prep[0, :])

    assert action.shape[0] == actions.shape[1]


def test_td_falcon_get_probabilistic_action(td_falcon_model):
    # Test the get_probabilistic_action method of TD_FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = td_falcon_model.prepare_data(states, actions, rewards)

    td_falcon_model.partial_fit(states_prep, actions_prep, rewards_prep)

    action = td_falcon_model.get_probabilistic_action(states_prep[0, :])

    assert isinstance(action.tolist(), float)


def test_td_falcon_get_rewards(td_falcon_model):
    # Test the get_rewards method of TD_FALCON
    states = np.random.rand(10, 2)
    actions = np.random.rand(10, 2)
    rewards = np.random.rand(10, 1)

    # Prepare data
    states_prep, actions_prep, rewards_prep = td_falcon_model.prepare_data(states, actions, rewards)

    td_falcon_model.partial_fit(states_prep, actions_prep, rewards_prep)

    predicted_rewards = td_falcon_model.get_rewards(states_prep, actions_prep)

    assert predicted_rewards.shape == rewards.shape
