import numpy as np

''' This class represents the exploratory epsilon-greedy policy
followed in generating the new samples stored in the replay memory.
It requires an estimator to predict the probabilities of the next
action and the number of available actions. The get_action() method
takes as input an observation of the environment and return the probability
for each possible action. The parameter epsilon allows the use of decaying
epsilon, so that the exploration can be reduced over time. '''
class EpsilonGreedyPolicy:
    def __init__(self, estimator, num_actions, input_channels, input_dim1, input_dim2):
        self.estimator = estimator
        self.num_actions = num_actions
        self.input_channels = input_channels
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
    def get_action(self, observation, epsilon):
        action_probs = np.ones(self.num_actions, dtype = np.float) * epsilon / self.num_actions
        q_values = self.estimator.predict(observation.reshape(1, self.input_channels, self.input_dim1, self.input_dim2))
        best_action = np.argmax(q_values)
        action_probs[best_action] += (1 - epsilon)
        return action_probs
