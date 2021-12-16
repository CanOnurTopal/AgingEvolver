from random import randrange


class MutatorHandler:
    """
    A framework agnostic implementation of a container class for Model mutating callbacks.
    """
    def __init__(self, mutators = None):
        """
        Args:
            mutators(list[function], function, optional): The mutator(s) are/is added to the MutatorHandler so they can be chosen at runtime.
        Raises:
            ValueError: Raised if a mutator object is not callable.
        """
        self._mutators = []
        if mutators is not None:
            self.add_mutator(mutators)

    def mutate(self, model):
        """
        Mutates the given model with a randomly chosen mutator.

        Args:
            model: The model that will be mutated.

        Returns:
            The mutated model.
        """
        index = randrange(len(self._mutators))
        return self._mutators[index](model)

    def get_mutator(self):
        """
        Returns random mutator function


        Returns:
            Mutator function.
        """

    def add_mutator(self, mutators):
        """
        Adds the callback(s) into the MutatorHandler's internal list to be utilised at runtime.

        Args:
            mutators(list[function], function, optional): The mutator(s) are/is added to the MutatorHandler so they can be chosen at runtime.
        Raises:
            ValueError: Raised if a mutator object is not callable.
        """
        try:
            #Check if mutator is iterable
            iterator = iter(mutators)
            for item in iterator:
                self.add_mutator(item) #recurse to add items
        except TypeError:
            # not iterable
            if not callable(mutators):
                raise ValueError("Mutators must be callable")
            else:
                self._mutators.append(mutators)

    def __len__(self):
        """
            Returns:
                int: how many mutators are stored
        """
        return len(self._mutators)
