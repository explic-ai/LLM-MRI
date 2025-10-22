from abc import ABC, abstractmethod
from typing import Optional

class DimensionalityReduction(ABC):
    """
    Abstract base class for dimensionality reduction techniques.
    """

    def __init__(self, n_components:int, random_state:Optional[int] = None, gridsize: Optional[int] = 10):
        """
        Initialize the dimensionality reduction object.

        :param dataset: The input dataset as a pandas DataFrame.
        :param n_components: The number of components to reduce to.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.gridsize = gridsize

    @abstractmethod
    def get_hidden_states_reduction(dataset):
        """
        Abstract method to perform the dimensionality reduction of all hidden states. 
        Returns a reduced version of the dataset's activations to n_components as a dictionary of DataFrames.
        The naming of each layer should be 'hidden_state_number' (e.g., 'hidden_state_1', 'hidden_state_2', etc.).

        :return: Dictionary, where each reduced hidden state is represented as a tensor, and its keys are "hidden_state_x"

        """
        pass        

    @abstractmethod
    def get_reduction(dataset):
        """
        Simple reduction by passing a dataset and returning a reduced dataset with n_components components.
        As of now, this method is only used on the 2D graph implementation, to generate the grids.
        """
        pass
    
