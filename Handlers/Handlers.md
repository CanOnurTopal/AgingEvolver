# Handlers package

## Submodules

## Handlers.DataHandler module


### class Handlers.DataHandler.DataHandler(x: numpy.ndarray, y: numpy.ndarray, x_test: Optional[numpy.ndarray] = None, y_test: Optional[numpy.ndarray] = None, test_size: Optional[Union[int, float]] = None, batch_size: int = 64)
Bases: `object`

Tensorflow based datahandler class.

This class is optimized for consistent data retrieval of constant data. The data is split into batches and compatible with keras.
The class also ensures data seperation between training and test sets.


#### property batch_size()
The batch size for training


* **Type**

    int



#### property test_dataset()
The test dataset for use in model evaluation.


* **Type**

    tf.Dataset



#### static train_test_split(x, y, test_size: Union[float, int])
Static method used to split train and test data. This is automatically called from __init__.


* **Parameters**

    
    * **x** (*np.ndarray*) – The data set


    * **y** (*no.ndarrag*) – The label set


    * **test_size** (*int**, **float*) – Amount of test data split from x and y.
    If the value is over 1, this is the absolute value of the test data size.
    If test_size is between 0 and 1, it represent the portion of data that will be used for testing.



* **Returns**

    x_training_data, y_training_data, x_test_data, y_test_data



#### property training_dataset()
The training dataset for use in model fitting.


* **Type**

    tf.Dataset



#### property x_shape()
this is the individual data shape that a model using this dataset will recieve.


* **Type**

    tf.Tensorspec



#### property y_shape()
this is the individual label shape that a model using this dataset will recieve.


* **Type**

    tf.Tensorspec


## Handlers.MutatorHandler module


### class Handlers.MutatorHandler.MutatorHandler(mutators=None)
Bases: `object`

A framework agnostic implementation of a container class for Model mutating callbacks.


#### add_mutator(mutators)
Adds the callback(s) into the MutatorHandler’s internal list to be utilised at runtime.


* **Parameters**

    **mutators** (*list**[**function**]**, **function**, **optional*) – The mutator(s) are/is added to the MutatorHandler so they can be chosen at runtime.



* **Raises**

    **ValueError** – Raised if a mutator object is not callable.



#### get_mutator()
Returns random mutator function


* **Returns**

    Mutator function.



#### mutate(model)
Mutates the given model with a randomly chosen mutator.


* **Parameters**

    **model** – The model that will be mutated.



* **Returns**

    The mutated model.
