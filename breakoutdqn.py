import gym
from mydqn.DQN import DQN

BREAKOUT_VALID_ACTIONS = [0, 1, 2, 3]
env = gym.make("Breakout-v0")
dqn = DQN(env, BREAKOUT_VALID_ACTIONS, replay_memory_max_dim = 500000)
(episode_durations, episode_total_rewards) = dqn.train(epsilon_decay_steps = 500000, num_episodes = 10000)
