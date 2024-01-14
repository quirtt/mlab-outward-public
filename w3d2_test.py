import random
from typing import Tuple
from dataclasses import asdict
import torch as t
import numpy as np
import gym
from utils import allclose, report, allclose_scalar, assert_all_equal
from w3d2_utils import make_env, set_seed


def _random_experience(num_actions, observation_shape, num_environments):
    obs = np.random.randn(num_environments, *observation_shape)
    actions = np.random.randint(0, num_actions - 1, (num_environments,))
    rewards = np.random.randn(num_environments)
    dones = np.random.randint(0, 1, (num_environments,)).astype(bool)
    next_obs = np.random.randn(num_environments, *observation_shape)
    return (obs, actions, rewards, dones, next_obs)


@report
def test_replay_buffer_single(
    cls, buffer_size=5, num_actions=2, observation_shape=(4,), num_environments=1, seed=1, device=t.device("cpu")
):
    """If the buffer has a single experience, that experience should always be returned when sampling."""
    from w3d2_part2_dqn_solution import ReplayBuffer

    rb: ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
    exp = _random_experience(num_actions, observation_shape, num_environments)
    rb.add(*exp)
    for _ in range(10):
        actual = rb.sample(1, device)
        allclose(actual.observations, t.tensor(exp[0]))
        allclose(actual.actions, t.tensor(exp[1]))
        allclose(actual.rewards, t.tensor(exp[2]))
        assert_all_equal(actual.dones, t.tensor(exp[3]))
        allclose(actual.next_observations, t.tensor(exp[4]))


@report
def test_replay_buffer_deterministic(
    cls, buffer_size=5, num_actions=2, observation_shape=(4,), num_environments=1, device=t.device("cpu")
):
    """The samples chosen should be deterministic, controlled by the given seed."""
    from w3d2_part2_dqn_solution import ReplayBuffer

    for seed in [67, 88]:
        rb: ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
        rb2: ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
        for _ in range(5):
            exp = _random_experience(num_actions, observation_shape, num_environments)
            rb.add(*exp)
            rb2.add(*exp)

        # Sequence of samples should be identical (ensuring they use self.rng)
        for _ in range(10):
            actual = rb.sample(2, device)
            actual2 = rb2.sample(2, device)
            for v, v2 in zip(asdict(actual).values(), asdict(actual2).values()):
                allclose(v, v2)


@report
def test_replay_buffer_wraparound(
    cls, buffer_size=4, num_actions=2, observation_shape=(1,), num_environments=1, seed=3, device=t.device("cpu")
):
    """When the maximum buffer size is reached, older entries should be overwritten."""
    from w3d2_part2_dqn_solution import ReplayBuffer

    rb: ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
    for i in range(6):
        rb.add(
            np.array([[float(i)]]),
            np.array([i % 2]),
            np.array([-float(i)]),
            np.array([False]),
            np.array([[float(i) + 1]]),
        )
    # Should be [4, 5, 2, 3] in the observations buffer now
    unique_obs = rb.sample(1000, device).observations.flatten().unique()
    assert_all_equal(unique_obs, t.arange(2, 6, device=device))


@report
def test_epsilon_greedy_policy(epsilon_greedy_policy):
    import w3d2_part2_dqn_solution

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test_eps_greedy_policy") for _ in range(5)])

    num_observations = np.array(envs.single_observation_space.shape, dtype=int).prod()
    num_actions = envs.single_action_space.n
    q_network = w3d2_part2_dqn_solution.QNetwork(num_observations, num_actions)
    obs = t.randn((envs.num_envs, *envs.single_observation_space.shape))
    greedy_action = w3d2_part2_dqn_solution.epsilon_greedy_policy(envs, q_network, np.random.default_rng(0), obs, 0)

    def get_actions(epsilon, seed):
        set_seed(seed)
        soln_actions = w3d2_part2_dqn_solution.epsilon_greedy_policy(
            envs, q_network, np.random.default_rng(seed), obs, epsilon
        )
        set_seed(seed)
        their_actions = epsilon_greedy_policy(envs, q_network, np.random.default_rng(seed), obs, epsilon)
        return soln_actions, their_actions

    def are_both_greedy(soln_acts, their_acts):
        return np.array_equal(soln_acts, greedy_action) and np.array_equal(soln_acts, greedy_action)

    both_greedy = [are_both_greedy(*get_actions(0.1, seed)) for seed in range(100)]
    assert np.mean(both_greedy) >= 0.9

    both_greedy = [are_both_greedy(*get_actions(0.5, seed)) for seed in range(100)]
    assert np.mean(both_greedy) >= 0.5

    both_greedy = [are_both_greedy(*get_actions(1, seed)) for seed in range(1000)]
    assert np.mean(both_greedy) > 0 and np.mean(both_greedy) < 0.1
