import numpy as np
import torch
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid
from gym import spaces
import gym
from collections import deque
from math import pow


from tqc import DEVICE


LOG_STD_MIN_MAX = (-20, 2)


class RescaleAction(gym.ActionWrapper):
    def __init__(self, env, a, b):
        assert isinstance(env.action_space, spaces.Box), (
            "expected Box action space, got {}".format(type(env.action_space)))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
        self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
        self.action_space = spaces.Box(low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype)

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low)*((action - self.a)/(self.b - self.a))
        action = np.clip(action, low, high)
        return action


class Mlp(Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = relu(fc(h))
        output = self.last_fc(h)
        return output


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), gamma=0.99, n_episodes_to_store=50, q_g_rollout_length=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.gamma = gamma
        self.n_episodes_to_store = n_episodes_to_store
        self.q_g_rollout_length = q_g_rollout_length

        self.transition_names = ('state', 'action', 'next_state', 'reward', 'not_done', 'ep_end',
                                 'returns', 'ep_length', 'bs_multiplier')
        sizes = (state_dim, action_dim, state_dim, 1, 1, 1, 1, 1, 1)
        for name, size in zip(self.transition_names, sizes):
            setattr(self, name, np.empty((max_size, size)))

        self.last_episodes = deque()

    def add(self, state, action, next_state, reward, done, ep_end):
        values = (state, action, next_state, reward, 1. - done, ep_end)
        for name, value in zip(self.transition_names, values):
            getattr(self, name)[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if ep_end:
            res_idx = []    # indices to put into fresh replay buffer
            running_return = 0
            running_tail_return = 0
            was_timelimit = self.not_done[(self.ptr - 1) % self.max_size, 0] > 0.5
            for t in range(self.size):
                idx = (self.ptr - 1 - t) % self.max_size   # index of tuple in replay
                if t > 0 and (self.ep_end[idx, 0] > 0.5):
                    break                                  # got to the previous episode end
                running_return = self.reward[idx] + self.gamma * running_return       # running return at current index
                if t >= self.q_g_rollout_length:
                    running_tail_return = self.reward[(idx + self.q_g_rollout_length) % self.max_size] + self.gamma * running_tail_return
                self.returns[idx] = running_return - np.power(self.gamma, self.q_g_rollout_length) * running_tail_return       # only indices in between
                self.ep_length[idx] = t + 1

                is_enough_rollout = t + 1 > self.q_g_rollout_length
                if (was_timelimit and is_enough_rollout) or not was_timelimit:
                    res_idx.append(idx)
                    if is_enough_rollout:
                        self.bs_multiplier[idx] = 1.0
                    else:
                        self.bs_multiplier[idx] = 0.0
            if len(res_idx) > 5:
                self.last_episodes.append(np.array(res_idx, dtype='int32'))
                if len(self.last_episodes) > self.n_episodes_to_store:
                    self.last_episodes.popleft()

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        names = self.transition_names[:-2]
        return (torch.FloatTensor(getattr(self, name)[ind]).to(DEVICE) for name in names)

    def gather_returns(self, gamma, n_per_episode):
        selected_idx = []
        for ep_lst in self.last_episodes:
            selected_idx.append(np.random.choice(ep_lst, replace=True, size=n_per_episode))
        selected_idx = np.concatenate(selected_idx)
        return self.get_returns_by_idx(selected_idx, gamma)

    def gather_returns_uniform(self, gamma, n_per_episode):
        all_idx = np.concatenate(self.last_episodes)
        selected_idx = np.random.choice(all_idx, replace=True, size=n_per_episode * len(self.last_episodes))
        return self.get_returns_by_idx(selected_idx, gamma)

    def get_returns_by_idx(self, selected_idx, gamma):
        result_states = self.state[selected_idx]
        result_actions = self.action[selected_idx]
        result_bs_multiplier = self.bs_multiplier[selected_idx]
        result_bs_states = self.state[(selected_idx + self.q_g_rollout_length) % self.max_size]         # if bs_multiplier is 0 non-existing states should not matter
        result_returns = self.returns[selected_idx]
        return (torch.tensor(arr, dtype=torch.float32, device=DEVICE) for arr in
                [result_states, result_actions, result_returns, result_bs_states, result_bs_multiplier])


class Critic(Module):
    def __init__(self, state_dim, action_dim, n_nets):
        super().__init__()
        self.nets = []
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Mlp(state_dim + action_dim, [256, 256], 1)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


class Actor(Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.net = Mlp(state_dim, [256, 256], action_dim)

    def forward(self, obs):
        mean = self.net(obs)
        return torch.tanh(mean)

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(DEVICE)[None, :]
        action = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=DEVICE),
                                      torch.ones_like(self.normal_std, device=DEVICE))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh
