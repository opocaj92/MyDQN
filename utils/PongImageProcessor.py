import numpy as np

''' This static class is used to preprocess images coming from the pong
simulator provided by OpenAI gym. The process_image() method crops the
screen frame, downsaples it by a factor of 2 and then converts it to B/W
coloring by setting the whole background to black and only moving objects
(the puddles and the ball) to white. The combine_image() method returns the
difference of two preprocessed frames to provide information of the movement
and trajectories of objects between frames. '''
class PongImageProcessor:
    @staticmethod
    def process_image(I):
        I = I[35:195]
        I = I[::2,::2,0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float)
    @staticmethod
    def combine_images(I1, I2):
        return I1 - I2
