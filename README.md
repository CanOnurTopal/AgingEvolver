# AgingEvolver

This is an keras based implementation of the evolutionary learning algorithm as described in in the paper arXiv:1802.01548.
The class manages the instantiation, evaluation and evolution of the competing models and reuses weights to optimize training.

Include the AgingEvolver module from AgingEvolver.py to get started. There is an example implementation located in Sample.py


# AgingEvolver module


### class AgingEvolver.AgingEvolver(x: numpy.ndarray, y: numpy.ndarray, x_test: Optional[numpy.ndarray] = None, y_test: Optional[numpy.ndarray] = None, test_size: Optional[Union[int, float]] = None, population_size: int = 30, batch_size: int = 64, metric='accuracy', maximize_metric: bool = True, mutators: Optional[list] = None, output_layer: Optional[keras.engine.base_layer.Layer] = None)
Bases: `object`

Once the parameters are set, the run() method will evolve a population of models in order to create the optimal model for the task.


#### add_mutator(mutator: callable)
Add one or more mutators that will be randomly selected each cycle during evaluation


* **Parameters**

    **callback** (*function*) – The mutator function that will take a model as its argument and return a mutated model



* **Raises**

    **ValueError** – This function can raise an exception is the callback is malconfigured.



#### property best_model()
Returns the best model and the score of the model.


* **Type**

    Model, Score



#### property history()
Every model that was ever evaluated.
Empty if the run() method is never called


* **Type**

    `List` containing `Models`



#### property population()
List containing tuples of `(Model, Score)` of the population that remains after the cycles are run.
Empty if the run() method is never called


#### property population_size()
The size of the population during evaluation


* **Type**

    int



#### run(cycles: int = 0, epochs: int = 50, sample_size: Optional[int] = None, optimizer='adam', verbose: bool = False)
This method runs the evolution process as described in the paper arXiv:1802.01548.
If population does not exist, initializes the population, otherwise continues with the existing population.

All models are recorded into the history property and the population is retained after running.


* **Parameters**

    
    * **cycles** (*int*) – The number of cycles the evolutionary model will run.
    A value of 0 or less will only populate the population.


    * **epochs** (*int**, **optional*) – The epochs used in fitting. Defaults to 50


    * **sample_size** (*int**, **optional*) – The sample size of each cycle.
    Defaults to 10% of population size if possible or 1


    * **optimizer** (*str**, **Keras Optimizer**, **Optional*) – The optimizer used in the fitting process. Defaults to adam


    * **verbose** (*bool*) – Print model information into stdout



* **Returns**

    Best performing model, score of the model



* **Raises**

    
    * **RuntimeError** – If metric or loss is not set. Will also raise if there are no mutators.


    * **ValueError** – If sample size is not greater than 0 or if sample_size is greater than population_size



#### set_architecture_generator(callback, mutate=True)
The architecture generator is used to initialize the population for the first run.
The callback function will be called to generate each of the initial population.

If a generator is not set, generation will default to mutating the input layer

IMPORTANT: the input layer will be provided at runtime and should not be added by the generator
:param callback: A callable object that returns an output layer to initialize population.

> The output layer must NOT include an input layer.


* **Parameters**

    **mutate** (*bool*) – Defines whether a random mutator should be applied to the result of the callback. Defaults to True.



* **Raises**

    **ValueError** – Can raise error if callback is malconfigured.



#### set_loss(loss, loss_weights=None)
Set the loss function that will be used during evaluation.


* **Parameters**

    
    * **loss** (*List**, **Dict**, **Function**, **str*) – A loss function/identifier or a collection of loss function/identifiers that Keras may use to fit the model


    * **loss_weights** (*List**, **Dict**, **Optional*) – If more than one loss function is provided, the weights used to average their values
    Can be a Dict object with keys corresponding to loss if loss is also a Dict object
