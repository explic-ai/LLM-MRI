from abc import ABC, abstractmethod

class DimensionalityReduction(ABC):
    """
    Abstract base class for dimensionality reduction techniques.
    """

    def __init__(self, n_components:int):
        """
        Initialize the dimensionality reduction object.

        :param dataset: The input dataset as a pandas DataFrame.
        :param n_components: The number of components to reduce to.
        """
        self.n_components = n_components

    @abstractmethod
    def get_reduction(dataset):
        """
        Abstract method to perform dimensionality reduction. 
        Returns a reduced version of the dataset's activations to n_components as a dictionary of DataFrames.
        The naming of each layer should be 'hidden_state_number' (e.g., 'hidden_state_1', 'hidden_state_2', etc.).

        :return: Dataset's activations as a dictionary of pandas DataFrame (one for each layer).

        """
        pass        