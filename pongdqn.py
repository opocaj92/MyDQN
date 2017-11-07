from theano import tensor as T
import gym
from mydqn.DQN import DQN
from utils.PongImageProcessor import PongImageProcessor

x = T.tensor4("x")
y = T.matrix("y")
PONG_VALID_ACTIONS = [2, 3]
env = gym.make("Pong-v0")
dqn = DQN(x, y, env, PONG_VALID_ACTIONS, PongImageProcessor)
(episode_durations, episode_total_rewards) = dqn.train()
