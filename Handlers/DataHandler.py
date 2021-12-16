from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Union
import numpy as np

class DataHandler:
    """
    Tensorflow based datahandler class.

    This class is optimized for consistent data retrieval of constant data. The data is split into batches and compatible with keras.
    The class also ensures data seperation between training and test sets.
    """
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 x_test: np.ndarray = None,
                 y_test: np.ndarray = None,
                 test_size: Union[int, float] = None,
                 batch_size: int = 64
                 ):
        """
        Args:
            x (np.ndarray): the training data that the models will fit.
                If no test data is provided, test data will be split of as defined by test_size.
            y (np.ndarray): the training labels that the models will fit.
                If no test data is provided, test data will be split of as defined by test_size.
            x_test (np.ndarray, optional): The test data that the models will use.
                Will be populated from argument x if not defined.
            y_test (np.ndarray, optional): The test labels that the models will use.
                Will be populated from argument y if not defined.
            test_size (int, float, optional): Amount of test data split from x and y.
                This must be given if no testing data is provided.
                If the value is over 1, this is the absolute value of the test data size.
                If test_size is between 0 and 1, it represent the portion of data that will be used for testing.
                Ignored if testing data is given.
            batch_size (int, optional) The batch size that will be used. Defaults to 64.

        """
        if x_test is None or y_test is None:
            if test_size is None:
                raise ValueError("test_size must be set if test data is not given")
            else:
                x, x, y_train, y_test = train_test_split(x, y, test_size=test_size)

        self._training_dataset = tf.data.Dataset.from_tensor_slices((x,y)).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self._testing_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self._batch_size = batch_size

        return

    @staticmethod
    def train_test_split(x, y, test_size: Union[float, int]):
        """
        Static method used to split train and test data. This is automatically called from __init__.

        Args:
            x (np.ndarray): The data set
            y (no.ndarrag): The label set
            test_size (int, float): Amount of test data split from x and y.
                If the value is over 1, this is the absolute value of the test data size.
                If test_size is between 0 and 1, it represent the portion of data that will be used for testing.

        Returns:
            x_training_data, y_training_data, x_test_data, y_test_data
        """
        return train_test_split(x, y, test_size = test_size)

    @property
    def training_dataset(self):
        """
        tf.Dataset: The training dataset for use in model fitting.
        """
        return self._training_dataset

    @property
    def test_dataset(self):
        """
        tf.Dataset: The test dataset for use in model evaluation.
        """
        return self._testing_dataset

    @property
    def x_shape(self):
        """
        tf.Tensorspec: this is the individual data shape that a model using this dataset will recieve.
        """
        return self._training_dataset.element_spec[0].shape[1:]

    @property
    def y_shape(self):
        """
        tf.Tensorspec: this is the individual label shape that a model using this dataset will recieve.
        """
        return self._training_dataset.element_spec[1].shape[1:]

    @property
    def batch_size(self):
        """int: The batch size for training"""
        return self._batch_size