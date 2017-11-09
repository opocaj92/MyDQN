import numpy as np
import random
from AtariImageProcessor import AtariImageProcessor

''' This class represent the replay memory in which experience samples are
stored. It is instantiated by providing the memory max dimension. The initialize()
method instantiate init_size samples into the memory taken from the env following
policy with valid_actions and epsilon as parameters. The insert() method receives
the elements of a sample and stores the sample in the memory, eventually removing
an old sample if the memory is full. The sample() methods returns batch_size samples
randomly selected among those stored in memory. '''
class ReplayMemory:
    def __init__(self, replay_memory_max_dim):
        self.replay_memory_max_dim = replay_memory_max_dim
        self.pool = list()
        self.used_dim = 0
    def initialize(self, env, init_size, policy, valid_actions, epsilon, input_dim1, input_dim2, input_channels):
        observation = env.reset()
        observation = AtariImageProcessor.process_image(observation, input_dim1, input_dim2)
        observation = AtariImageProcessor.combine_images(observation, observation, input_channels)
        for _ in range(min(init_size, self.replay_memory_max_dim)):
            probs = policy.get_action(observation, epsilon)
            a = np.random.choice(np.arange(policy.num_actions), p = probs)
            next_observation, reward, done, _ = env.step(valid_actions[a])
            next_observation = AtariImageProcessor.process_image(next_observation, input_dim1, input_dim2)
            next_observation = AtariImageProcessor.combine_images(next_observation, observation, input_channels)
            self.pool.append((observation, a, reward, next_observation, done))
            self.used_dim += 1
            if done:
                observation = env.reset()
                observation = AtariImageProcessor.process_image(observation)
                observation = AtariImageProcessor.combine_images(observation, observation)
            else:
                observation = next_observation
    def insert(self, observation, a, reward, next_observation, done):
        if self.used_dim == self.replay_memory_max_dim:
            self.pool.pop(0)
            self.used_dim -= 1
        self.pool.append((observation, a, reward, next_observation, done))
        self.used_dim += 1
    def sample(self, batch_size):
        return random.sample(self.pool, batch_size)
