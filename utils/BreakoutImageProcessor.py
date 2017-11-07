import numpy as np
import scipy

''' This static class is used to preprocess images coming from the breakout
simulator provided by OpenAI gym. The process_image() method crops the
screen frame, convert it to grayscale and downsaples it using the nearest
neighbours interpolation. The combine_image() method returns the combination
of the previous observation, formed by four stacked images, with the new image,
by removing the first image from the observation and appending the new one. '''
class BreakoutImageProcessor:
    @staticmethod
    def process_image(I):
        I = I[35:195]
        I = np.dot(I, [0.2989, 0.5870, 0.1140])
        I = scipy.misc.imresize(I, (84, 84), "nearest")
        return I.astype(np.float)
    @staticmethod
    def combine_images(I1, I2):
        if len(I1.shape) == 3:
            return np.append(I1[:, :, 1:], np.expand_dims(I2, 2), axis = 2)
        else:
            return np.stack([I1] * 4, axis = 2)
