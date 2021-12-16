from typing import Union
from tensorflow import keras
from tensorflow.keras import layers, metrics
import numpy as np
from Handlers.DataHandler import DataHandler
from Handlers.MutatorHandler import MutatorHandler

class AgingEvolver:
    """
    This is an keras based implementation of the evolutionary learning algorithm as described in in the paper arXiv:1802.01548.
    The class manages the instantiation, evaluation and evolution of the competing models and reuses weights to optimize training.

    Once the parameters are set, the run() method will evolve a population of models in order to create the optimal model for the task.
    """
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 x_test: np.ndarray = None,
                 y_test: np.ndarray = None,
                 test_size: Union[int, float] = None,
                 population_size: int = 30,
                 batch_size: int = 64,
                 metric = "accuracy",
                 maximize_metric: bool = True,
                 mutators: list = None,
                 output_layer: layers.Layer = None
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
                If the value is over 1, this is the absolute value of the test data size.
                If test_size is between 0 and 1, it represent the portion of data that will be used for testing.
                Ignored if testing data is given.
            population_size (int, optional) the size of the competing population. Defaults to 30
            batch_size (int, optional) The batch size that will be used. Defaults to 64.
            metric (keras.metrics.Metric, str) the keras metric used to evalueate the models. Defaults to "accuracy"
            maximize_metric (bool) defines if larger metric value is better. Defaults to True.
            mutators (List[function]): The mutators that will be randomly selected to mutate a model at each cycle.
            output_layer (keras.layers.Layer, Optional) The output layer of the model. Will default to a softmax dense layer.
        """
        self._data: DataHandler = DataHandler(x, y, x_test=x_test, y_test=y_test, test_size=test_size, batch_size=batch_size)
        self._mutators: MutatorHandler = MutatorHandler(mutators)
        self._loss = None
        self._loss_weights = None
        self._metric: str = metric
        self._maximize_metric: bool = bool(maximize_metric)
        self._architecture_callback: callable = None
        self._architecture_mutate: bool = True
        self._population_size: int = int(population_size)
        if self._population_size < 2:
            raise ValueError("Population size cannot be less than 2")
        self._history: list = [] #Model container
        self._performance_history: np.ndarray = np.zeros(0, dtype=float)
        self._population: np.ndarray = np.arange(self._population_size, dtype=int) #The population indices in the history
        self.__oldest_pop: int = 0
        if output_layer is None:
            self._output_layer = layers.Dense(self._data.y_shape[0])
        else:
            self._output_layer = output_layer
        self._input_layer = layers.Input(shape=self._data.x_shape, name="Input_Layer")
        return


    def add_mutator(self, mutator: callable):
        """Add one or more mutators that will be randomly selected each cycle during evaluation

        Args:
            callback (function): The mutator function that will take a model as its argument and return a mutated model

        Raises:
            ValueError: This function can raise an exception is the callback is malconfigured.
        """
        self._mutators.add_mutator(mutator)
        return


    def set_loss(self, loss, loss_weights = None):
        """Set the loss function that will be used during evaluation.

        Args:
            loss(List, Dict, Function, str): A loss function/identifier or a collection of loss function/identifiers that Keras may use to fit the model
            loss_weights(List, Dict, Optional): If more than one loss function is provided, the weights used to average their values
                Can be a Dict object with keys corresponding to loss if loss is also a Dict object
        """
        self._loss = loss
        self._loss_weights = loss_weights
        return

    def run(self, cycles: int = 0, epochs : int = 50, sample_size: int = None, optimizer="adam", verbose: bool = False):
        """
        This method runs the evolution process as described in the paper arXiv:1802.01548.
        If population does not exist, initializes the population, otherwise continues with the existing population.

        All models are recorded into the history property and the population is retained after running.

        Args:
            cycles (int): The number of cycles the evolutionary model will run.
                A value of 0 or less will only populate the population.
            epochs (int, optional): The epochs used in fitting. Defaults to 50
            sample_size (int, optional): The sample size of each cycle.
                Defaults to 10% of population size if possible or 1
            optimizer (str, Keras Optimizer, Optional): The optimizer used in the fitting process. Defaults to adam
            verbose (bool): Print model information into stdout

        Returns:
            Best performing model, score of the model

        Raises:
            RuntimeError: If metric or loss is not set. Will also raise if there are no mutators.
            ValueError: If sample size is not greater than 0 or if sample_size is greater than population_size
        """
        if sample_size is None: #Set default sample_size
            sample_size = max(int(self._population_size * 0.1), 1)
        self._run_validation(sample_size)
        cycles += len(self._history)  # Ensure that the cycles that have already occured are included in the cycle count
        if len(self._performance_history) < cycles:
            #Append empty array to prevent costly resizing operations.
            reserved_length = max(cycles, self._population_size) - len(self._performance_history)
            dtype = self._performance_history.dtype
            self._performance_history = np.append(self._performance_history, np.empty(reserved_length, dtype=dtype))
        #initialize population
        while len(self._history) < self._population_size:
            new_model = self._architecture_generator()
            self._insert_new_model(new_model, epochs, optimizer, verbose=verbose)
        rng = np.random.default_rng(cycles + 1)  # random number generator seeded with the cycle count
        while len(self._history) < cycles:
            #Generate an array of unique and random indices for population sampling
            sample_indices = rng.choice(self._population_size, sample_size, replace=False)
            sample_set = self._population[sample_indices]
            if self._maximize_metric:
                best_model_index = np.argmax(self._performance_history[sample_set])
            else:
                best_model_index = np.argmin(self._performance_history[sample_set])
            loop_count = 0
            while loop_count < 100:
                try:
                    new_model = self._mutators.mutate(self._history[best_model_index])
                    if self._insert_new_model(new_model, epochs, optimizer, verbose=verbose):
                        break
                    loop_count += 1
                    continue
                except ValueError:
                    loop_count += 1
                    continue
            if loop_count >= 100:
                raise RuntimeWarning("Mutators not enough or layers provided by mutators are not compatible with each other."
                                     "Provide additional mutators")
                return self.best_model
        return self.best_model

    def set_architecture_generator(self, callback, mutate=True):
        """
        The architecture generator is used to initialize the population for the first run.
        The callback function will be called to generate each of the initial population.

        If a generator is not set, generation will default to mutating the input layer

        IMPORTANT: the input layer will be provided at runtime and should not be added by the generator
        Args:
            callback (callable): A callable object that returns an output layer to initialize population.
                The output layer must NOT include an input layer.
            mutate (bool): Defines whether a random mutator should be applied to the result of the callback. Defaults to True.

        Raises:
            ValueError: Can raise error if callback is malconfigured.
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self._architecture_callback = callback
        self._architecture_mutate = bool(mutate)
        return

    def _run_validation(self, sample_size):
        """
        Ensures that the object is in a correct state and validates arguments when run() method is called

        Args:
             sample_size (int): sample_size parameter forwarded from run() method
        """
        if len(self._mutators) == 0:
            raise RuntimeError("No mutators have been set prior to run")
        if self._loss is None:
            raise RuntimeError("No loss function has been set")
        if self._metric is None:
            raise RuntimeError("No metric has been set")
        if int(sample_size) <= 0 or int(sample_size) >= self._population_size:
            raise ValueError("sample_size must be greater than 0 and less than the population size")
        return

    def _architecture_generator(self):
        """
        Wrapping Function for the default/user provided architecture generator that generates the initial population.

        Returns:
            Keras Layer: Keras layer that includes the Input layer
        """
        if self._architecture_callback is not None:
            arch = self._architecture_callback()
            arch = arch(self._input_layer)
        else:
            arch = self._input_layer
        if self._architecture_mutate:
            arch = self._mutators.mutate(arch)
        return arch

    def _insert_new_model(self, model, epochs, optimizer, verbose=False):
        """
        Evaluates the model and inserts the model into self._history and the score into self._performance_history
        Calls self._insert_pop with the _history index of the model

        Args:
            model (Keras Layers): the model that will be inserted into population and history
            epochs (int): The epochs passed to the fitting function
            optimizer (Keras Optimizer, str) The optimizer passed to the fitting function
            verbose (bool): Print model information into stdout

        Returns:
            True if insertion occured, false otherwise
        """
        index = len(self._history)
        model_result = self._fit_evaluate_model(model, epochs, optimizer, verbose=verbose)
        if model_result is None:
            return False
        model, score = model_result
        self._history.append(model)
        self._performance_history[index] = score
        self._insert_pop(index)
        return True

    def _insert_pop(self, model_index: int):
        """
        Inserts population into self._population and removes oldest pop from population

        Args:
            model_index (int): The index of the model inside self._history
        """
        self._population[self.__oldest_pop] = model_index
        self.__oldest_pop = (self.__oldest_pop + 1) % self._population_size
        return


    def _fit_evaluate_model(self, layer, epochs, optimizer, verbose=False):
        """
        Args:
            layer (Keras Layers): the model that will be inserted into population and history
            epochs (int): The epochs passed to the fitting function
            optimizer (Keras Optimizer, str) The optimizer passed to the fitting function
            verbose (bool): Print model information into stdout

        Returns:
            Fitted model layers (without the output layer), score
            None if generated model cannot be utilized due to incompatible layers
        """
        layer = layers.Flatten()(layer)
        layer = layers.Dense(self._data.y_shape[0])(layer)
        model = keras.Model(inputs=self._input_layer, outputs=layer)
        model.compile(loss=self._loss, optimizer=optimizer, metrics=[self._metric])
        try:
            history = model.fit(self._data.training_dataset, validation_data=self._data.test_dataset,
                            epochs=epochs, batch_size=self._data.batch_size)
        except ValueError:
            return None
        if verbose:
            model.summary()
        if isinstance(self._metric, str):
            metric_name = self._metric
        else:
            metric_name = self._metric.name
        return model.layers[-3].output, history.history[metric_name][0]

    @property
    def population_size(self):
        """int: The size of the population during evaluation"""
        return self._population_size

    @property
    def population(self):
        """List containing tuples of :obj:`(Model, Score)` of the population that remains after the cycles are run.
        Empty if the run() method is never called"""
        processed = []
        if len(self._history) < self._population_size:
            # Return if run is never called
            return processed
        for index in self._population:
            processed.append(
                (self._history[index], self._performance_history[index])
            )
        return processed

    @property
    def history(self):
        """:Tuple:`List` containing :obj:`Models`: Every model that was ever evaluated.
        Empty if the run() method is never called"""
        processed = []
        for index in range(len(self._history)): #inefficient but returning parallel arrays would not be pythonic
            processed.append(
                (self._history[index], self._performance_history[index])
            )
        return processed

    @property
    def best_model(self):
        """Model, Score: Returns the best model and the score of the model."""
        if self._maximize_metric:
            index = np.argmax(self._performance_history)
        else:
            index = np.argmin(self._performance_history)
        return self._history[index], self._performance_history[index]




if __name__ == "__main__":
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
    a = AgingEvolver(x_train, y_train, x_test=x_test, y_test=y_test, population_size=10)
    a.set_loss('categorical_crossentropy')
    a.add_mutator([layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), layers.Conv2D(64, kernel_size=(3, 3), activation="relu")])
    a.run(5,5,1)