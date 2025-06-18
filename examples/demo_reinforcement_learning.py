import numpy as np
from tqdm import tqdm
from artlib import TD_FALCON, FuzzyART, complement_code
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
from copy import deepcopy

STATE_MAX = 47
ACTION_MAX = 3
REWARD_MAX = 150

# only works with Fuzzy ART based FALCON
def prune_clusters(cls):
    # get existing state and action weights
    state = np.array(cls.fusion_art.modules[0].W)
    action = np.array(cls.fusion_art.modules[1].W)
    # get predicted rewards for each cluster
    reward = np.array(cls.fusion_art.get_channel_centers(2)).reshape((-1, 1))

    # combine state and actions
    combined = np.round(np.hstack((state, action)), decimals=3)
    # identify unique combinations
    unique_combined, indices = np.unique(combined, axis=0, return_inverse=True)
    # get mean reward prediction for each unique state-action pair
    unique_rewards = np.array(
        [reward[indices == i, :].mean() for i in range(len(unique_combined))]
    )

    # split unique state-action pairs
    unique_states = unique_combined[:, : state.shape[1]]
    unique_actions = unique_combined[:, state.shape[1] :]

    # update model to only have unique state-action pairs and their average rewards
    cls.fusion_art.modules[0].W = [row for row in unique_states]
    cls.fusion_art.modules[1].W = [row for row in unique_actions]
    cls.fusion_art.modules[2].W = [
        row for row in complement_code(unique_rewards.reshape((-1, 1)))
    ]

    return cls


def update_FALCON(records, cls, shrink_ratio):
    # convert records into arrays
    states = np.array(records["states"]).reshape((-1, 1))
    actions = np.array(records["actions"]).reshape((-1, 1))
    rewards = np.array(records["rewards"]).reshape((-1, 1))

    # complement code states and actions
    states_cc = complement_code(states)
    actions_cc = complement_code(actions)
    rewards_cc = complement_code(rewards)

    # states_fit, actions_fit, sarsa_rewards_fit = cls.calculate_SARSA(states_cc, actions_cc, rewards_cc, single_sample_reward=1.0)

    # if FALCON has been previously trained
    if hasattr(cls.fusion_art.modules[0], "W"):
        # remove any duplicate clusters
        cls = prune_clusters(cls)
        # # shrink clusters to account for dynamic programming changes
        # for m in range(3):
        #     cls.fusion_art.modules[m] = cls.fusion_art.modules[m].shrink_clusters(shrink_ratio)

    # fit FALCON to data
    # data = cls.fusion_art.join_channel_data([states_fit, actions_fit, sarsa_rewards_fit])
    # cls.fusion_art = cls.fusion_art.partial_fit(data)
    cls = cls.partial_fit(
        states_cc, actions_cc, rewards_cc, single_sample_reward=1.0
    )
    for m in range(3):
        cls.fusion_art.modules[m] = cls.fusion_art.modules[m].shrink_clusters(
            shrink_ratio
        )

    return cls


def training_cycle(
    cls,
    epochs,
    steps,
    sarsa_alpha,
    sarsa_gamma,
    render_mode=None,
    shrink_ratio=0.1,
    explore_rate=0.0,
    checkpoint_frequency=50,
    early_stopping_reward=-20,
):
    # create the environment
    env = gym.make("CliffWalking-v0", render_mode=render_mode)
    # define some constants
    ACTION_SPACE = np.array([[0], [1.0], [2.0], [3.0]])
    STATE_MAX = 47
    ACTION_MAX = 3
    REWARD_MAX = 150
    cls.td_alpha = sarsa_alpha
    cls.td_lambda = sarsa_gamma

    best_reward = -np.inf
    best_cls = None

    # track reward history
    reward_history = []

    pbar = tqdm(range(epochs))
    for e in pbar:
        observation, info = env.reset()
        n_observation = observation / STATE_MAX
        records = {"states": [], "actions": [], "rewards": []}
        past_states = []
        for _ in range(steps):
            # get an action
            observation_cc = complement_code(
                np.array([n_observation]).reshape(1, -1)
            )
            if np.random.random() < explore_rate:
                action = int(np.random.choice(ACTION_SPACE.flatten()))
            else:
                cls_action = cls.get_action(
                    observation_cc, action_space=ACTION_SPACE, optimality="min"
                )
                action = int(cls_action[0])
            # normalize state and action
            n_observation = observation / STATE_MAX
            n_action = action / ACTION_MAX

            # record state and action for training
            records["states"].append(n_observation)
            records["actions"].append(n_action)
            past_states.append(observation)

            # take a step
            observation, reward, terminated, truncated, info = env.step(action)

            # check reward value
            if reward > -100:
                # punish circular paths
                if observation in past_states:
                    reward = -2
            # normalize and record reward from step
            n_reward = abs(reward) / REWARD_MAX
            records["rewards"].append(n_reward)

            # check if epoch is done
            if terminated or truncated or reward == -100:
                break

        # train FALCON
        cls = update_FALCON(records, cls, shrink_ratio)
        # record sum of rewards generated during this epoch
        reward_history.append(-sum(records["rewards"]) * REWARD_MAX)

        # if this isnt random exploration
        if explore_rate < 1.0:
            # see if we should save a checkpoint
            if (e + 1) % checkpoint_frequency == 0 or e == epochs - 1:
                # test model
                cls, test_reward_history = demo_cycle(
                    cls, 1, steps, render_mode=None
                )
                # check if our current model is better than the best previous model
                if test_reward_history[0] >= best_reward or best_cls is None:
                    # same a checkpoint
                    best_cls = deepcopy(cls)
                    best_reward = test_reward_history[0]
                    # check early stopping condition
                    if best_reward > early_stopping_reward:
                        # show current best reward on progress bar
                        pbar.set_postfix({"Best Reward": best_reward})
                        return cls, reward_history
                else:
                    # restore previous best model
                    cls = deepcopy(best_cls)
            # show current best reward on progress bar
            pbar.set_postfix({"Best Reward": best_reward})

    env.close()
    return cls, reward_history


def demo_cycle(cls, epochs, steps, render_mode=None):
    # create the environment
    env = gym.make("CliffWalking-v0", render_mode=render_mode)
    # define some constants
    ACTION_SPACE = np.array([[0], [1.0], [2.0], [3.0]])

    # track reward history
    reward_history = []

    for _ in range(epochs):
        observation, info = env.reset()
        n_observation = observation / STATE_MAX
        records = {"states": [], "actions": [], "rewards": []}
        past_states = []
        for _ in range(steps):
            # get an action
            observation_cc = complement_code(
                np.array([n_observation]).reshape(1, -1)
            )
            cls_action = cls.get_action(
                observation_cc, action_space=ACTION_SPACE, optimality="min"
            )
            action = int(cls_action[0])

            # normalize state and action
            n_observation = observation / STATE_MAX
            n_action = action / ACTION_MAX

            # record state and action
            records["states"].append(n_observation)
            records["actions"].append(n_action)
            past_states.append(observation)

            # take a step
            observation, reward, terminated, truncated, info = env.step(action)

            # check reward value
            if reward > -100:
                # punish circular paths
                if observation in past_states:
                    reward = -2
            # normalize and record reward from step
            n_reward = abs(reward) / REWARD_MAX
            records["rewards"].append(n_reward)

            # check if epoch is done
            if terminated or truncated or reward == -100:
                break
        # record sum of rewards generated during this epoch
        reward_history.append(-sum(records["rewards"]) * REWARD_MAX)

    env.close()
    return cls, reward_history


def max_up_to_each_element(lst):
    max_list = []
    current_max = float("-inf")  # Start with the smallest possible value
    for num in lst:
        current_max = max(current_max, num)
        max_list.append(current_max)
    return max_list


def train_FALCON():
    # define training regimen
    training_regimen = [
        {
            "name": "random",
            "epochs": 1000,
            "shrink_ratio": 0.3,
            "gamma": 0.0,
            "explore_rate": 1.0,
            "render_mode": None,
        },
        {
            "name": "explore 33%",
            "epochs": 500,
            "shrink_ratio": 0.3,
            "gamma": 0.2,
            "explore_rate": 0.333,
            "render_mode": None,
        },
        {
            "name": "explore 20%",
            "epochs": 500,
            "shrink_ratio": 0.3,
            "gamma": 0.2,
            "explore_rate": 0.20,
            "render_mode": None,
        },
        {
            "name": "explore 5%",
            "epochs": 1000,
            "shrink_ratio": 0.3,
            "gamma": 0.2,
            "explore_rate": 0.05,
            "render_mode": None,
        },
    ]
    MAX_STEPS = 25
    SARSA_ALPHA = 1.0

    # define parameters for state, action, and reward modules
    art_state = FuzzyART(rho=0.99, alpha=0.01, beta=1.0)
    art_action = FuzzyART(rho=0.99, alpha=0.01, beta=1.0)
    art_reward = FuzzyART(rho=0.0, alpha=0.01, beta=1.0)
    art_state.set_data_bounds(np.zeros((1,)), np.ones((1,)))
    art_action.set_data_bounds(np.zeros((1,)), ACTION_MAX*np.ones((1,)))
    art_reward.set_data_bounds(np.zeros((1,)), np.ones((1,)))
    # instantiate FALCON
    cls = TD_FALCON(art_state, art_action, art_reward, channel_dims=[2, 2, 2])
    # record rewards for each epoch
    all_rewards = []
    # initialize FALCON with random exploration
    print("Starting Training")
    for regimen in training_regimen:
        print(f"Starting Training Cycle: {regimen['name']}")
        cls, reward_history = training_cycle(
            cls,
            epochs=regimen["epochs"],
            steps=MAX_STEPS,
            sarsa_alpha=SARSA_ALPHA,
            sarsa_gamma=regimen["gamma"],
            render_mode=regimen["render_mode"],
            shrink_ratio=regimen["shrink_ratio"],
            explore_rate=regimen["explore_rate"],
        )
        all_rewards.extend(reward_history)
        print("MAX REWARD:", max(reward_history))

    # demo learned policy
    cls, reward_history = demo_cycle(
        cls, epochs=2, steps=25, render_mode="human"
    )
    print(reward_history)
    all_rewards.extend(reward_history)
    max_rewards = max_up_to_each_element(all_rewards)

    # plot reward history
    plt.figure()
    plt.plot(list(range(len(all_rewards))), all_rewards, "r-")
    plt.plot(list(range(len(all_rewards))), max_rewards, "b-")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Rewards over Time")
    plt.show()


if __name__ == "__main__":
    # This takes approximately 3 minutes
    np.random.seed(42)
    train_FALCON()
