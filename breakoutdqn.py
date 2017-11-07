from theano import tensor as T
import gym
from mydqn.DQN import DQN
from utils.BreakoutImageProcessor import BreakoutImageProcessor

x = T.tensor4("x")
y = T.matrix("y")
BREAKOUT_VALID_ACTIONS = [0, 1, 2, 3]
env = gym.make("Breakout-v0")
dqn = DQN(x, y, env, BREAKOUT_VALID_ACTIONS, BreakoutImageProcessor, input_channels = 4, input_dim1 = 84, input_dim2 = 84)
(episode_durations, episode_total_rewards) = dqn.train()
