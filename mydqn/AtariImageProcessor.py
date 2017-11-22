import numpy as np
import scipy

''' This static class is used to preprocess images coming from the Atari
simulators provided by OpenAI gym. The process_image() method crops the
screen frame, convert it to grayscale and downsaples it using the nearest
neighbours interpolation. The combine_image() method returns the combination
of the previous observation, formed by four stacked images, with the new image,
by removing the first image from the observation and appending the new one. '''
class AtariImageProcessor:
    @staticmethod
    def process_image(I, img_size1 = 84, img_size2 = 84):
        I = I[34:194]
        I = np.round(np.dot(I, [0.2989, 0.587, 0.114])).astype(np.int)
        I = scipy.misc.imresize(I, (img_size1, img_size2), "nearest")
        return I.astype(np.float)
    @staticmethod
    def combine_images(I1, I2, m = 4):
        if len(I1.shape) == 3 and I1.shape[0] == m:
            return np.append(I1[1:, :, :], np.expand_dims(I2, 0), axis = 0)
        else:
            return np.stack([I1] * m, axis = 0)
