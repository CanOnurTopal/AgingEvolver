from tensorflow import keras
from tensorflow.keras import layers, metrics
import numpy as np
from AgingEvolver import AgingEvolver


"""
This example uses the example dataset provided by keras and solves it using evolutionary learning without human input.
"""

def cifar_solver():
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    a = AgingEvolver(x_train, y_train, x_test=x_test, y_test=y_test, metric="accuracy")

    a.set_loss('categorical_crossentropy')

    a.add_mutator([layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                   layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                   layers.Conv2D(120, kernel_size=(3, 3), activation="relu"),
                   layers.MaxPooling2D(pool_size=(2, 2)),
                   layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
                   layers.Conv2D(8, kernel_size=(3, 3), activation="relu")])

    model, score = a.run(cycles=100, epochs=100, verbose=True)
    print("Score:", score)



if __name__ == "__main__":
    cifar_solver()