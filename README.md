# MyDQN

## A simple implementation of the Deep Q-Network used to play Atari games

### How it works
This is an implementation of the famous Deep Q-Network (DQN) proposed by Google DeepMind Labs to play Atari game from raw pixel, realized using Theano+Lasagne for the neural networks and OpenAI gym for the reinforcement learning environment. The code is structured as a library, aiming at being reusable for different applications than the two proposed examples (Pong and Breakout).
The *utils* directory contains helper functions to manipulate raw pixel frames (resizing, converting to grayscale or similar) for the two proposed examples, while the *mydqn* folder contains the real DQN implementation.
- *ReplayMemory.py* implements the experience replay mechanism used by the DQN to break the statistichal dependency of consecutive samples.
- *EpsilonGreedyPolicy.py* implements the exploratory policy followed when generating new samples.
- *ConvNN.py* implements the convolutional neural network (CNN) used to represent both the target network and the Q-network of the DQN.
- *DQN.py* joins together the other components to implement the DQN itself and its training process. Its many parameters should allow for flexibility and reusability in a variety of context.
The model itself is extremely memory consuming and CPU intensive, so it is probably going to get killed by your system if you don't execute it on a powerful machine (ideally on a GPU). This project was done as an exercise and as a simple example of this model, and doesn't aim to good performances or great scalability.

### Author
*Castellini Jacopo*
