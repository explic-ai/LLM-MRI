from abc import ABC, abstractmethod
from typing import List, Union

class Graph(ABC):
    """
    Abstract base class for building the graph accordingly to the dimensionality reduction type and number of components.
    """

    def __init__(self, n_components:int, categories: dict):
        """
        Initialize the graph object.

        :param n_components: The number of components that the graph was reduced based on.
        """
        self.n_components = n_components

    @abstractmethod
    def build_graph(dataset):
        """

        """
        pass        

        