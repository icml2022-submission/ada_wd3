import numpy as np
import torch
from math import ceil, floor

from tqc.functions import quantile_huber_loss_f
from tqc import DEVICE


class Trainer(object):
	def __init__(
		self,
		*,
		actor,
		critic,
		critic_target,
		discount,
		tau,
		beta,
		beta_lr,
		sampling_scheme,
		target_policy_noise,
		noise_clip,
		policy_freq,
		Q_G_eval_interval,
		Q_G_n_per_episode,
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target

		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount
		self.tau = tau

		self.log_beta = torch.tensor(np.log(beta / (1 - beta)), requires_grad=True, device=DEVICE)
		self.beta_lr = beta_lr
		self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.beta_lr)

		self.total_it = 0
		self.Q_G_delta = torch.zeros((1,), requires_grad=False, device=DEVICE)
		self.sampling_scheme = sampling_scheme
		self.Q_G_eval_interval = Q_G_eval_interval
		self.Q_G_n_per_episode = Q_G_n_per_episode

		self.target_policy_noise = target_policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

	def train(self, replay_buffer, batch_size=256,):
		metrics = dict()
		state, action, next_state, reward, not_done, *_ = replay_buffer.sample(batch_size)
		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action = self.actor(next_state)
			noise = (
				torch.randn_like(new_next_action) * self.target_policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			new_next_action = (new_next_action + noise).clamp(-1, 1)

			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)  # batch x nets x 1
			next_z = next_z.squeeze()
			metrics['Target_Q/Q_value'] = next_z.mean().__float__()

			# compute target
			raw_target = reward + not_done * self.discount * next_z
			beta = self.log_beta.sigmoid()
			min_target = raw_target.min(dim=1, keepdim=True)[0]
			mean_target = raw_target.mean(dim=1, keepdim=True)
			target = min_target * beta + mean_target * (1 - beta)

		cur_z = self.critic(state, action).squeeze()
		critic_loss = ((cur_z - target)**2).mean()
		metrics['critic_loss'] = critic_loss.item()

		# --- Policy and alpha loss ---
		new_action = self.actor(state)
		actor_loss = - self.critic(state, new_action).mean(2).mean(1, keepdim=True).mean()
		metrics['actor_loss'] = actor_loss.item()

		# --- Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if self.total_it % self.policy_freq == 0:
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

		beta = self.log_beta.sigmoid()
		beta_loss = - beta * self.Q_G_delta.detach()
		metrics['beta'] = beta.item()
		metrics['beta_loss'] = beta_loss.item()
		metrics['Q_G_delta'] = self.Q_G_delta.item()

		# -- eta loss
		if self.total_it > 10000 and self.total_it % self.Q_G_eval_interval == 0:
			self.eval_thresholds(replay_buffer, self.Q_G_n_per_episode)
			beta = self.log_beta.sigmoid()
			beta_loss = - beta * self.Q_G_delta.detach()
			self.beta_optimizer.zero_grad()
			beta_loss.backward()
			self.beta_optimizer.step()

		self.total_it += 1
		return metrics

	def eval_thresholds_by_type(self, replay_buffer, n_per_episode, sampling_scheme):
		res = dict()
		if sampling_scheme == 'uniform':
			states, actions, returns, bs_states, bs_multiplier = replay_buffer.gather_returns_uniform(self.discount, n_per_episode)
		elif sampling_scheme == 'episodes':
			states, actions, returns, bs_states, bs_multiplier = replay_buffer.gather_returns(self.discount, n_per_episode)
		else:
			raise Exception("No such sampling scheme")
		tail_actions = self.actor(bs_states)
		tail_z = self.critic_target(bs_states, tail_actions)
		tail_z = tail_z.reshape(tail_z.shape[0], -1)
		tail_z = tail_z.mean(1, keepdim=True) * bs_multiplier * np.power(replay_buffer.gamma, replay_buffer.q_g_rollout_length)
		res[f'LastReplay_{sampling_scheme}/Returns'] = (returns + tail_z).mean().__float__()

		cur_z = self.critic(states, actions)
		res[f'LastReplay_{sampling_scheme}/Q_value'] = cur_z.mean().__float__()
		return res

	def eval_thresholds(self, replay_buffer, n_per_episode):
		res_uniform = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'uniform')
		res_episodes = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'episodes')
		res = dict()
		res.update(res_uniform)
		res.update(res_episodes)
		self.Q_G_delta[0] = res[f'LastReplay_{self.sampling_scheme}/Q_value'] - \
							res[f'LastReplay_{self.sampling_scheme}/Returns']
		return res

	def save(self, filename):
		filename = str(filename)
		self.light_save(filename)
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.beta_optimizer.state_dict(), filename + "_beta_optimizer")

	def light_save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.log_beta, filename + "_log_beta")

	def load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.log_beta = torch.load(filename + "_log_beta")
		self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.beta_lr)
		self.beta_optimizer.load_state_dict(torch.load(filename + "_beta_optimizer"))
