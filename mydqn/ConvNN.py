import numpy as np
import theano
from theano import tensor as T
import lasagne

''' This class represent a convolutional neural network, that can be used
to implement both the target network and the Q-network of the DQN. To be
initialized it requires various parameters, like x and y, two Theano symbolic
variables representing respectively the input batch and the expected target batch,
the dimensions of the input image frame, the number of possible actions of the
output, the learning rate and decaying factor of the RMSProp learning algorithm.
The get_params_value() and set_params_value() methods are used in the DQN parameters
copy step, while the save_model() and load_model() methods can be used to save
the trained network to a file. '''
class ConvNN:
    def __init__(self, input_channels, input_dim1, input_dim2, num_actions, learning_rate, rho, epsilon):
        x = T.tensor4("x")
        y = T.vector("y")
        a = T.ivector("a")
        self.l_input = lasagne.layers.InputLayer((None, input_channels, input_dim1, input_dim2), x)
        self.l_conv1 = lasagne.layers.Conv2DLayer(self.l_input, 32, (8, 8), stride = 4, nonlinearity = lasagne.nonlinearities.rectify)
        self.l_conv2 = lasagne.layers.Conv2DLayer(self.l_conv1, 64, (4, 4), stride = 2, nonlinearity = lasagne.nonlinearities.rectify)
        self.l_conv3 = lasagne.layers.Conv2DLayer(self.l_conv2, 64, (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
        self.l_dense1 = lasagne.layers.DenseLayer(self.l_conv3, 512)
        self.l_dense2 = lasagne.layers.DenseLayer(self.l_dense1, num_actions, nonlinearity = lasagne.nonlinearities.softmax)
        self.out = lasagne.layers.get_output(self.l_dense2)
        self.values = self.out.take(a, axis = 1)
        self.loss = T.mean(lasagne.objectives.squared_error(self.values, y))
        self.params = lasagne.layers.get_all_params(self.l_dense2, trainable = True)
        self.updates = lasagne.updates.rmsprop(self.loss, self.params, learning_rate = learning_rate, rho = rho, epsilon = epsilon)
        self.train_fn = theano.function([x, y, a], self.loss, updates = self.updates, allow_input_downcast = True)
        self.predict = theano.function([x], self.out, allow_input_downcast = True)
    def get_params_value(self):
        return lasagne.layers.get_all_param_values(self.l_dense2)
    def set_params_value(self, params):
        lasagne.layers.set_all_param_values(self.l_dense2, params)
    def save_model(self, name):
        np.save(name, lasagne.layers.get_all_param_values(self.l_dense2))
    def load_model(self, name):
        lasagne.layers.set_all_param_values(self.l_dense2, np.load(name))
