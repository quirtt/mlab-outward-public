import torch as t
import numpy as np
import gym
from dataclasses import asdict
from utils import allclose, report, allclose_scalar, assert_all_equal
from w3d2_utils import make_env, set_seed


@report
def test_agent_init(agent_class):
    import w3d3_solution

    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "run name I guess") for i in range(4)])
    expected_agent = w3d3_solution.Agent(envs)
    actual_agent = agent_class(envs)

    expected_params = sum(p.nelement() for p in expected_agent.parameters())
    actual_params = sum(p.nelement() for p in actual_agent.parameters())
    assert expected_params == actual_params

    # All biases are 0
    assert all([(p == 0).all().item() for name, p in actual_agent.named_parameters() if "bias" in name])

    # Row norms are correct
    def check_row_norm(layer, expected):
        assert layer.weight.norm(dim=-1).allclose(t.full((layer.weight.shape[0],), expected)) or layer.weight.T.norm(
            dim=-1
        ).allclose(t.full((layer.weight.T.shape[0],), expected))

    check_row_norm(actual_agent.actor[4], 0.01)
    check_row_norm(actual_agent.critic[4], 1.0)
    for layer in [actual_agent.actor[0], actual_agent.actor[2], actual_agent.critic[0], actual_agent.critic[2]]:
        check_row_norm(layer, t.sqrt(t.tensor([2])).item())


@report
def test_compute_advantages(compute_advantages):
    def test_perfect_value():
        rewards = t.tensor([1.0, 2.0, 3.0]).unsqueeze(dim=1)  # r_0, ... r_2
        values = t.tensor([1 + 2 + 3 + 10, 2 + 3 + 10, 3 + 10]).unsqueeze(dim=1)  # V(s_0) ... V(s_2)
        dones = t.tensor([0, 0, 0]).unsqueeze(dim=1)
        next_done = t.tensor([0])
        next_value = t.tensor([10]).unsqueeze(dim=1)  # V(s_3)
        gamma = 1
        gae_lambda = 1
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, "cpu", gamma, gae_lambda)
        assert advantages[0] == 0
        assert advantages[1] == 0
        assert advantages[2] == 0

    def test_zero_value():
        rewards = t.tensor([1, 2, 3]).unsqueeze(dim=1)  # r_0, ... r_2
        values = t.tensor([0, 0, 0]).unsqueeze(dim=1)  # V(s_0) ... V(s_2)
        dones = t.tensor([0, 0, 0]).unsqueeze(dim=1)
        next_done = t.tensor([0])
        next_value = t.tensor([0]).unsqueeze(dim=1)  # V(s_3)
        gamma = 1
        gae_lambda = 1
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, "cpu", gamma, gae_lambda)
        assert advantages[0] == rewards[2] + rewards[1] + rewards[0]
        assert advantages[1] == rewards[2] + rewards[1]
        assert advantages[2] == rewards[2]

    def test_batched():
        rewards = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).T  # r_0, ... r_2
        values = t.tensor([[0, 0, 0], [0, 0, 0]]).T  # V(s_0) ... V(s_2)
        dones = t.tensor([[0, 0, 0], [0, 0, 0]]).T
        next_done = t.tensor([0, 0])
        next_value = t.tensor([[0, 0]])  # V(s_3)
        gamma = 1
        gae_lambda = 1
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, "cpu", gamma, gae_lambda)
        assert t.isclose(advantages[0], rewards[2] + rewards[1] + rewards[0]).all()
        assert t.isclose(advantages[1], rewards[2] + rewards[1]).all()
        assert t.isclose(advantages[2], rewards[2]).all()

    def test_dones():
        rewards = t.tensor([[1, 2, 3], [1, 2, 3]]).T  # r_0, ... r_2
        values = t.tensor([[1 + 2, 2, 3 + 10], [1 + 2 + 3, 2 + 3, 3]]).T  # V(s_0) ... V(s_2)
        dones = t.tensor([[0, 0, 1], [0, 0, 0]]).T
        next_done = t.tensor([0, 1])
        next_value = t.tensor([[10, 17]])  # V(s_3)
        gamma = 1
        gae_lambda = 1
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, "cpu", gamma, gae_lambda)
        assert (advantages == 0).all()

    def test_gae_0():
        rewards = t.tensor([1, 2, 3]).unsqueeze(dim=1)  # r_0, ... r_2
        values = t.tensor([0, 0, 0]).unsqueeze(dim=1)  # V(s_0) ... V(s_2)
        dones = t.tensor([0, 0, 0]).unsqueeze(dim=1)
        next_done = t.tensor([0])
        next_value = t.tensor([0]).unsqueeze(dim=1)  # V(s_3)
        gamma = 1
        gae_lambda = 0
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, "cpu", gamma, gae_lambda)
        assert advantages[0] == rewards[0]
        assert advantages[1] == rewards[1]
        assert advantages[2] == rewards[2]

    def test_gae_lambda():
        rewards = t.tensor([1.0, 2.0, 4.0]).unsqueeze(dim=1)  # r_0, ... r_2
        values = t.tensor([0, 0, 0]).unsqueeze(dim=1)  # V(s_0) ... V(s_2)
        dones = t.tensor([0, 0, 0]).unsqueeze(dim=1)
        next_done = t.tensor([0])
        next_value = t.tensor([0]).unsqueeze(dim=1)  # V(s_3)
        gamma = 1
        gae_lambda = 0.5
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, "cpu", gamma, gae_lambda)
        assert t.isclose(advantages[0], rewards[0] + gae_lambda * rewards[1] + gae_lambda**2 * rewards[2])
        assert t.isclose(advantages[1], rewards[1] + gae_lambda * rewards[2])
        assert t.isclose(advantages[2], rewards[2])

    def test_gae_lambda_with_value():
        rewards = t.tensor([1.0, 2.0, 3.0]).unsqueeze(dim=1)  # r_0, ... r_2
        values = t.tensor([0, 1, 2]).unsqueeze(dim=1)  # V(s_0) ... V(s_2)
        dones = t.tensor([0, 0, 0]).unsqueeze(dim=1)
        next_done = t.tensor([0])
        next_value = t.tensor([0]).unsqueeze(dim=1)  # V(s_3)
        gamma = 1
        gae_lambda = 0.5
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, "cpu", gamma, gae_lambda)
        assert t.isclose(
            advantages[0],
            (
                rewards[0]
                + values[1]
                - values[0]
                + gae_lambda * (values[2] - values[1] + rewards[1])
                + gae_lambda**2 * (next_value - values[2] + rewards[2])
            ),
        )
        assert t.isclose(
            advantages[1], (values[2] - values[1] + rewards[1] + gae_lambda * (next_value - values[2] + rewards[2]))
        )
        assert t.isclose(advantages[2], next_value - values[2] + rewards[2])

    def test_gamma_and_lambda():
        rewards = t.tensor([1.0, 2.0, 3.0]).unsqueeze(dim=1)  # r_0, ... r_2
        values = t.tensor([0, 1, 2]).unsqueeze(dim=1)  # V(s_0) ... V(s_2)
        dones = t.tensor([0, 0, 0]).unsqueeze(dim=1)
        next_done = t.tensor([0])
        next_value = t.tensor([0]).unsqueeze(dim=1)  # V(s_3)
        gamma = 0.5
        gae_lambda = 0.5
        advantages = compute_advantages(next_value, next_done, rewards, values, dones, "cpu", gamma, gae_lambda)
        assert t.isclose(
            advantages[0],
            (
                rewards[0]
                + gamma * values[1]
                - values[0]
                + gamma * gae_lambda * (gamma * values[2] - values[1] + rewards[1])
                + gamma**2 * gae_lambda**2 * (gamma * next_value - values[2] + rewards[2])
            ),
        )
        assert t.isclose(
            advantages[1],
            (
                gamma * values[2]
                - values[1]
                + rewards[1]
                + gamma * gae_lambda * (gamma * next_value - values[2] + rewards[2])
            ),
        )
        assert t.isclose(advantages[2], gamma * next_value - values[2] + rewards[2])

    test_perfect_value()
    test_zero_value()
    test_batched()
    test_dones()
    test_gae_0()
    test_gae_lambda()
    test_gae_lambda_with_value()
    test_gamma_and_lambda()


@report
def test_minibatch_indexes(minibatch_indexes):
    batch_size, minibatch_size = 1024, 128
    indexes = minibatch_indexes(batch_size, minibatch_size)

    assert len(indexes) == batch_size // minibatch_size
    assert all([batch_inds.shape == (minibatch_size,) for batch_inds in indexes])
    assert np.concatenate(indexes).tolist() != list(range(batch_size))
    assert np.sort(np.concatenate(indexes)).tolist() == list(range(batch_size))


@report
def test_make_minibatches(make_minibatches):

    n_steps, n_envs = 10, 4
    batch_size, minibatch_size = n_steps * n_envs, 5
    obs_shape, action_shape = tuple(), tuple()

    get_tensor = lambda: t.arange(0, batch_size).reshape((n_steps, n_envs))
    args = [get_tensor() for _ in range(5)] + [obs_shape, action_shape, batch_size, minibatch_size]
    minibatches = make_minibatches(*args)
    keys = ["obs", "logprobs", "actions", "advantages", "values"]

    # Shapes are right
    assert all([all([asdict(mb)[key].shape == (minibatch_size,) for key in keys]) for mb in minibatches])

    # Chosen steps are consistent within each minibatch
    for mb in minibatches:
        vals = t.stack([asdict(mb)[key] for key in keys])
        assert (vals[0] == vals).all()

    # All steps are present
    all_obs = t.cat([mb.obs for mb in minibatches]).sort().values
    assert t.equal(all_obs, t.arange(0, batch_size))

    # Returns are correct
    assert all([t.equal(mb.returns, mb.advantages + mb.values) for mb in minibatches])


@report
def test_calc_policy_loss(calc_policy_loss):
    import w3d3_solution

    def run_test(target_ratio):
        probs = t.tensor([[0.1, 0.9], [0.1, 0.9]])
        probs = t.distributions.Categorical(probs)
        mb_action = t.tensor([0, 0])
        mb_advantages = t.tensor([-1, 1], dtype=t.float)
        mb_logprobs = t.full_like(mb_advantages, -2.3026)
        mb_logprobs = (mb_logprobs.exp() / target_ratio).log()
        clip_coef = 0.1
        args = probs, mb_action, mb_advantages, mb_logprobs, clip_coef
        expected_loss = w3d3_solution.calc_policy_loss(*args).item()
        actual_loss = calc_policy_loss(*args).item()
        allclose_scalar(actual_loss, expected_loss)

    for ratio in t.linspace(0.05, 2, 40):
        run_test(ratio)


@report
def test_calc_value_function_loss(calc_value_function_loss):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    class FakeCritic(t.nn.Module):
        def forward(self, x):
            return t.tensor([[-1.0, 1.0]]).T

    critic = t.nn.Sequential(FakeCritic()).to(device)
    mb_obs = t.randn((2, 4))
    mb_returns = t.tensor([-2.0, 0.0])
    loss = calc_value_function_loss(critic, mb_obs, mb_returns, 0.1)
    assert loss == 0.05


@report
def test_calc_entropy_loss(calc_entropy_loss):
    for _ in range(5):
        probs = t.distributions.Categorical(logits=t.randn((128, 2)))
        ent_coef = np.random.random()
        actual_loss = calc_entropy_loss(probs, ent_coef)
        expected_loss = -ent_coef * probs.entropy().mean()
        allclose_scalar(actual_loss, expected_loss)


@report
def test_ppo_scheduler(ppo_scheduler_class):
    inital_lr, end_lr, num_updates = 1.1, 0.1, 10
    optimizer = t.optim.SGD([t.nn.Parameter(t.tensor([0.0]))], 1)
    ppo_scheduler = ppo_scheduler_class(optimizer, inital_lr, end_lr, num_updates)
    for expected_lr in t.linspace(inital_lr, end_lr, num_updates + 1)[:-1]:
        ppo_scheduler.step()
        actual_lr = optimizer.param_groups[0]["lr"]
        allclose_scalar(actual_lr, expected_lr.item())
