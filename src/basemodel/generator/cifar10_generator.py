def _blur(image):
    import cv2 as cv
    import random
    return cv.GaussianBlur(image, ksize=(0, 0), sigmaX=random.randint(1, 3), sigmaY=0)


def cifar10_generators(type, batch_size=32, val_split=0.15):
    from keras.preprocessing.image import ImageDataGenerator
    from keras.datasets import cifar10
    from sklearn.model_selection import train_test_split

    if type == "train":
        (X_train, _), (_, _) = cifar10.load_data()
        X_train = X_train / 255.

        validation_split = val_split
        X_train, X_val = train_test_split(X_train, test_size=validation_split)

        train_gen = cifar10_generator_transform_x_and_y(X_train, batch_size, shuffle=True)
        val_gen = cifar10_generator_transform_x_and_y(X_val, batch_size, shuffle=True)

        return (train_gen, val_gen)

    elif type == "test":
        # We don't want random transformation on test set
        (_, _), (X_test, _) = cifar10.load_data()
        X_test = X_test / 255.
        base_args = {
            "preprocessing_function": _blur
        }
        return ImageDataGenerator(**base_args).flow(X_test, X_test, batch_size=batch_size, shuffle=False)


import numpy as np
from keras.utils import Sequence


class cifar10_generator_transform_x_and_y(Sequence):
    def __init__(self, X, batch_size, shuffle=False):
        self.X = X
        self.batch_size = batch_size
        self.indices = np.arange(self.X.shape[0])
        self.shuffle = shuffle
        # otherwise i have error in base class
        self.n = X.shape[0]
        self.next_index = 0

    def __getitem__(self, index):
        import tensorflow as tf

        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Apply random transformation
        batch = np.array(
            [
                tf.image.random_flip_left_right(
                    tf.image.random_flip_up_down(
                        self.X[index])) for index in inds
            ])
        batch_blurred = np.array([_blur(image) for image in batch])

        # return X,y
        return batch_blurred, batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        from math import floor
        return floor(self.X.shape[0] / self.batch_size)

    def next(self):
        batch_blurred, batch = self.__getitem__(self.next_index)
        self.next_index += 1
        return batch_blurred, batch


class cifar10_generator_two_inputs(Sequence):

    def __init__(self, X, batch_size, shuffle=False):
        self.X = X
        self.batch_size = batch_size
        self.indices = np.arange(self.X.shape[0])
        self.shuffle = shuffle
        # otherwise i have error in base class
        self.num_batches = X.shape[0] // self.batch_size
        self.next_index = 0

    def __getitem__(self, index):
        import tensorflow as tf

        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch = np.array(self.X[inds])
        batch_blurred = np.array([_blur(image) for image in batch])

        batch_resized = np.array(tf.image.resize(batch, [16, 16]))
        batch_resized_blurred = np.array(tf.image.resize(batch_blurred, [16, 16]))

        # return X,y
        return [batch_resized_blurred, batch_blurred], [batch_resized, batch]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        from math import floor
        return floor(self.X.shape[0] / self.batch_size)

    def next(self):
        [batch_resized_blurred, batch_blurred], [batch_resized, batch] = self.__getitem__(self.next_index)
        self.next_index += 1
        return [batch_resized_blurred, batch_blurred], [batch_resized, batch]
