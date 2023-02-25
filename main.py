import os
import gym
import d4rl
import time
import torch
import utils
import argparse
import numpy as np


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment 
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="hopper-medium-v0")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5, type=float)
	parser.add_argument("--use_normalization", default=True)
	parser.add_argument("--gpu", default='0', type=str)
	# Changes
	parser.add_argument("--mix_env", default='', type=str)
	parser.add_argument("--mix_ratio", default=0.0, type=float)
	parser.add_argument("--coef_grad_random_action", default=0.0, type=float)
	parser.add_argument("--k", default=1.0, type=float)

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	
	file_name = f"{args.policy}-{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha,
		# Gradient Penalty
		'coef_grad_random_action' : args.coef_grad_random_action,
		'k': args.k,
	}

	# Initialize policy
	import TD3_BC
	policy = TD3_BC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	# We flag expert and non-expert demonstrations, for diagnostic purposes only.
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env), flag='expert')
	if args.mix_env != '':
		random_replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
		random_env = gym.make(args.mix_env)
		random_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(random_env), flag='random')
		random_size = int(replay_buffer.size * args.mix_ratio)
		expert_size = replay_buffer.size - random_size
		# Constructing the expert-cloned datasets. Trajectories of the BC agents are located in the second half.
		if 'cloned' in args.mix_env:
			replay_buffer.state = np.vstack((replay_buffer.state[:expert_size], random_replay_buffer.state[::-1][:random_size]))
			replay_buffer.action = np.vstack((replay_buffer.action[:expert_size], random_replay_buffer.action[::-1][:random_size]))
			replay_buffer.next_state = np.vstack((replay_buffer.next_state[:expert_size], random_replay_buffer.next_state[::-1][:random_size]))
			replay_buffer.reward = np.vstack((replay_buffer.reward[:expert_size], random_replay_buffer.reward[::-1][:random_size]))
			replay_buffer.not_done = np.vstack((replay_buffer.not_done[:expert_size], random_replay_buffer.not_done[::-1][:random_size]))
			replay_buffer.source = np.vstack((replay_buffer.source[:expert_size], random_replay_buffer.source[::-1][:random_size]))
			replay_buffer.size = replay_buffer.state.shape[0]
		# For the expert-random datasets,
		else:
			replay_buffer.state = np.vstack((replay_buffer.state[:expert_size], random_replay_buffer.state[:random_size]))
			replay_buffer.action = np.vstack((replay_buffer.action[:expert_size], random_replay_buffer.action[:random_size]))
			replay_buffer.next_state = np.vstack((replay_buffer.next_state[:expert_size], random_replay_buffer.next_state[:random_size]))
			replay_buffer.reward = np.vstack((replay_buffer.reward[:expert_size], random_replay_buffer.reward[:random_size]))
			replay_buffer.not_done = np.vstack((replay_buffer.not_done[:expert_size], random_replay_buffer.not_done[:random_size]))
			replay_buffer.source = np.vstack((replay_buffer.source[:expert_size], random_replay_buffer.source[:random_size]))
			replay_buffer.size = replay_buffer.state.shape[0]
		print('total exp ', replay_buffer.size, \
				'expert exp ratio ', expert_size/(expert_size + random_size), \
					'non-expert exp ratio ', random_size/(expert_size + random_size))

	if args.use_normalization:
		mean, std = replay_buffer.normalize_states() 
		print('normalizaed ')
	else:
		mean,std = 0,1
		print('not normalizaed ')
	
	begin_time = time.time()
	for t in range(int(args.max_timesteps)):
		policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1} for {time.time() - begin_time} s")
			begin_time = time.time()
			evaluation = eval_policy(policy, args.env, args.seed, mean, std)
			policy.diag(replay_buffer, evaluation, f"./results/{file_name}")
			if args.save_model: policy.save(f"./models/{file_name}")
