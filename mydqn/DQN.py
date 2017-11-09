import numpy as np
import itertools
import os.path
from ConvNN import ConvNN
from ReplayMemory import ReplayMemory
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from AtariImageProcessor import AtariImageProcessor

''' This class represent the deep Q-Network itself. It is initialized with many
parameters used to create properly the experience replay memory and the two networks
used by the training algorithm. The copy_params() methods simply updates the
parameters of the target_network with those of the q_network. The train() method
is the one that properly train the whole DQN. '''
class DQN:
    def __init__(self, env, valid_actions, input_channels = 4, input_dim1 = 84, input_dim2 = 84, learning_rate = 25e-5, rho = 0.95, epsilon = 0.01, replay_memory_max_dim = 1000000):
        print "Creating target network and Q-network..."
        self.target_network = ConvNN(input_channels, input_dim1, input_dim2, len(valid_actions), learning_rate, rho, epsilon)
        self.q_network = ConvNN(input_channels, input_dim1, input_dim2, len(valid_actions), learning_rate, rho, epsilon)
        print "Creating experience replay memory..."
        self.replay_memory = ReplayMemory(replay_memory_max_dim)
        self.env = env
        self.valid_actions = valid_actions
        self.input_channels = input_channels
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        if os.path.isfile("target_network.npy") and os.path.isfile("q_network.npy"):
            print "Loading networks parameters from file..."
            self.target_network.load_model("target_network.npy")
            self.q_network.load_model("q_network.npy")
    def copy_params(self):
        params = self.q_network.get_params_value()
        self.target_network.set_params_value(params)
    def train(self, init_size = 50000, max_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_steps = 1000000, update_every = 4, copy_every = 10000, discount_factor = 0.99, num_episodes = 100000, batch_size = 32):
        policy = EpsilonGreedyPolicy(self.q_network, len(self.valid_actions), self.input_channels, self.input_dim1, self.input_dim2)
        epsilons = np.linspace(max_epsilon, min_epsilon, epsilon_decay_steps)
        print "Initializing the replay memory..."
        self.replay_memory.initialize(self.env, init_size, policy, self.valid_actions, max_epsilon, self.input_dim1, self.input_dim2, self.input_channels)
        global_step = 0
        update_step = 0
        loss = 0
        episode_durations = [0 for _ in range(num_episodes)]
        episode_total_rewards = [0 for _ in range(num_episodes)]
        print "Start training..."
        for episode in range(num_episodes):
            observation = self.env.reset()
            observation = AtariImageProcessor.process_image(observation, self.input_dim1, self.input_dim2)
            observation = AtariImageProcessor.combine_images(observation, observation, self.input_channels)
            for t in itertools.count():
                epsilon = epsilons[min(global_step, epsilon_decay_steps - 1)]
                probs = policy.get_action(observation, epsilon)
                a = np.random.choice(np.arange(policy.num_actions), p = probs)
                next_observation, reward, done, _ = self.env.step(self.valid_actions[a])
                next_observation = AtariImageProcessor.process_image(next_observation, self.input_dim1, self.input_dim2)
                next_observation = AtariImageProcessor.combine_images(next_observation, observation, self.input_channels)
                episode_durations[episode] = t
                episode_total_rewards[episode] += reward
                self.replay_memory.insert(observation, a, reward, next_observation, done)
                if t % update_every == 0:
                    if update_step % copy_every == 0:
                        print "Copying parameters to target network..."
                        self.copy_params()
                    samples = self.replay_memory.sample(batch_size)
                    observations_batch, actions_batch, rewards_batch, next_observations_batch, done_batch = map(np.array, zip(*samples))
                    q_next_values = self.target_network.predict(next_observations_batch)
                    target_values = rewards_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_next_values, axis = 1)
                    loss = self.q_network.train_fn(observations_batch, target_values, actions_batch)
                    update_step += 1
                global_step += 1
                if done:
                    break
                observation = next_observation
            print "Episode {}/{} ended in {} steps with a total reward of {}, loss = {}".format(episode, num_episodes, episode_durations[episode], episode_total_rewards[episode], loss)
        print "Training ended!"
        print "Saving the networks to a file..."
        self.target_network.save_model("target_network")
        self.q_network.save_model("q_network")
        return (episode_durations, episode_total_rewards)
