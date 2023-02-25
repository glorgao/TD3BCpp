import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GP_FREQUENTCY = 5

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3_BC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		coef_grad_random_action=0.0,
		k=0.0
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha
		self.coef_grad_random_action = coef_grad_random_action
		self.k = k

		self.total_it = 0
		self.logger = []

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done, source = replay_buffer.sample(batch_size)
		
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise  
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# TD3BC++ adds a Gradient Penalty (+GP) module,
		if self.total_it % GP_FREQUENTCY == 0:
			gradient_norm_wrt_random_action = self.penalize_gradient_norm(state)
			critic_loss = critic_loss +  gradient_norm_wrt_random_action * self.coef_grad_random_action

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()
			
			# TD3BC++ adds a Critic Weighted Constraint Relaxation (+CR) module,
			current_Q = ((current_Q1 + current_Q2) * 0.5).squeeze()
			# we add a clamp for Walker2d-Medium-Expert task as we find there exists a long-tail Q-value distribution:
			# _min, _max = np.percentile(current_Q.cpu().detach().numpy(), q=[5, 95])
			# current_Q = current_Q.clamp(_min, _max)
			
			current_Q = (current_Q - current_Q.min()) / (current_Q.max() - current_Q.min())
			actor_loss = -lmbda * Q.mean() + (F.mse_loss(pi, action, reduction='none').mean(axis=-1) * current_Q.detach()).mean()
			# actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action).mean()

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
	
	def penalize_gradient_norm(self, state):
		_state_rep = state.clone().detach().repeat(16, 1).requires_grad_(True)
		_random_action = torch.rand(size=self.actor(_state_rep).size(), requires_grad=True) * 2 - 1.0
		_random_action= _random_action.clamp(-self.max_action, self.max_action).to(device)
		current_Q1, current_Q2 = self.critic(_state_rep, _random_action)
		grad_q1_random_action = torch.autograd.grad(
			outputs=current_Q1.sum(),
			inputs =_random_action,
			create_graph=True
		)[0]
		grad_q2_random_action = torch.autograd.grad(
			outputs=current_Q2.sum(),
			inputs =_random_action,
			create_graph=True
		)[0]
		grad_q1_random_action = torch.norm(grad_q1_random_action, p=2, dim=-1)
		grad_q2_random_action = torch.norm(grad_q2_random_action, p=2, dim=-1)
		grad_q_random_action = F.relu(grad_q1_random_action - self.k) **2  + F.relu(grad_q2_random_action - self.k) **2
		grad_q_random_action = grad_q_random_action.mean()

		return grad_q_random_action

	def monitor_gradient_norm(self, replay_buffer):
		state, action, next_state, reward, not_done, source = replay_buffer.sample(5120)
		_state_rep = state.clone().detach().repeat(16, 1).requires_grad_(True)
		_random_action = torch.rand(size=self.actor(_state_rep).size(), requires_grad=True) * 2 - 1.0
		_random_action= _random_action.clamp(-self.max_action, self.max_action).to(device)
		current_Q1, current_Q2 = self.critic(_state_rep, _random_action)
		_, grad_q1_random_action = torch.autograd.grad(
			outputs=current_Q1.sum(),
			inputs=(_state_rep, _random_action),
			create_graph=True
		)
		_, grad_q2_random_action = torch.autograd.grad(
			outputs=current_Q2.sum(),
			inputs=(_state_rep, _random_action),
			create_graph=True
		)
		grad_q1_random_action = torch.norm(grad_q1_random_action, p=2, dim=-1)
		grad_q2_random_action = torch.norm(grad_q2_random_action, p=2, dim=-1)
		grad_q_random_action = (grad_q1_random_action + grad_q2_random_action) /2
		grad_q_random_action = grad_q_random_action.cpu().detach().numpy()

		return list(np.percentile(grad_q_random_action, q=[100, 90, 50, 0]))

	def diag(self, replay_buffer, evaluation, file_name):
		tmp = self.monitor_gradient_norm(replay_buffer)
		tmp.append(evaluation*100)
		self.logger.append(tmp)
		
		import pandas as pd 
		to_csv_file = pd.DataFrame(self.logger)
		to_csv_file.columns = ['GN_max', 'GN_p90', 'GN_med', 'GN_min', 'eval']
		print(to_csv_file.round(1))
		to_csv_file.to_csv(file_name + '.csv')
