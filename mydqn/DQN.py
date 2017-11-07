import numpy as np
import itertools
import os.path
from ConvNN import ConvNN
from ReplayMemory import ReplayMemory
from EpsilonGreedyPolicy import EpsilonGreedyPolicy

''' This class represent the deep Q-Network itself. It is initialized with many
parameters used to create properly the experience replay memory and the two networks
used by the training algorithm. The copy_params() methods simply updates the
parameters of the target_network with those of the q_network. The train() method
is the one that properly train the whole DQN. '''
class DQN:
    def __init__(self, x, y, env, valid_actions, image_processor, input_channels = 1, input_dim1 = 80, input_dim2 = 80, learning_rate = 1e-2, rho = 0.99, replay_memory_max_dim = 500000):
        print "Creating target network and Q-network..."
        self.target_network = ConvNN(x, y, input_channels, input_dim1, input_dim2, len(valid_actions), learning_rate, rho)
        self.q_network = ConvNN(x, y, input_channels, input_dim1, input_dim2, len(valid_actions), learning_rate, rho)
        print "Creating experience replay memory..."
        self.replay_memory = ReplayMemory(replay_memory_max_dim)
        self.env = env
        self.valid_actions = valid_actions
        self.image_processor = image_processor
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
    def train(self, init_size = 50000, max_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_steps = 500000, copy_every = 10000, discount_factor = 0.99, num_episodes = 100000, batch_size = 32):
        policy = EpsilonGreedyPolicy(self.q_network, len(self.valid_actions), self.input_channels, self.input_dim1, self.input_dim2)
        epsilons = np.linspace(max_epsilon, min_epsilon, epsilon_decay_steps)
        print "Initializing the replay memory..."
        self.replay_memory.initialize(self.env, self.image_processor, init_size, policy, self.valid_actions, max_epsilon)
        global_train_step = 0
        episode_durations = [0 for _ in range(num_episodes)]
        episode_total_rewards = [0 for _ in range(num_episodes)]
        print "Start training..."
        for episode in range(num_episodes):
            observation = self.env.reset()
            observation = self.image_processor.process_image(observation)
            observation = self.image_processor.combine_images(observation, observation)
            for t in itertools.count():
                if global_train_step % copy_every == 0:
                    print "Copying parameters to target network..."
                    self.copy_params()
                epsilon = epsilons[min(global_train_step, epsilon_decay_steps - 1)]
                probs = policy.get_action(observation, epsilon)
                a = np.random.choice(np.arange(policy.num_actions), p = probs)
                next_observation, reward, done, _ = self.env.step(self.valid_actions[a])
                next_observation = self.image_processor.process_image(next_observation)
                next_observation = self.image_processor.combine_images(next_observation, observation)
                episode_durations[episode] = t
                episode_total_rewards[episode] += reward
                self.replay_memory.insert(observation, a, reward, next_observation, done)
                samples = self.replay_memory.sample(batch_size)
                observations_batch, actions_batch, rewards_batch, next_observations_batch, done_batch = map(np.array, zip(*samples))
                observations_batch = observations_batch.reshape(batch_size, self.input_channels, self.input_dim1, self.input_dim2)
                next_observations_batch = next_observations_batch.reshape(batch_size, self.input_channels, self.input_dim1, self.input_dim2)
                q_next_values = self.target_network.predict(next_observations_batch)
                target_values = rewards_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_next_values, axis = 1)
                labels = np.zeros((batch_size, len(self.valid_actions)))
                for n, (a, v) in enumerate(zip(actions_batch, target_values)):
                    labels[n][a] = v
                loss = self.q_network.train_fn(observations_batch, labels)
                if done:
                    break
                observation = next_observation
                global_train_step += 1
            print "Episode {}/{} ended in {} steps with a total reward of {}".format(episode, num_episodes, episode_durations[episode], episode_total_rewards[episode])
        print "Training ended!"
        print "Saving the networks to a file..."
        self.target_network.save_model("target_network")
        self.q_network.save_model("q_network")
        return (episode_durations, episode_total_rewards)
