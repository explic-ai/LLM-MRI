from ._base import DimensionalityReduction
from torchdr import PCA as TorchPCA
import pandas as pd

class PCA(DimensionalityReduction):
    """
    Principal Component Analysis (PCA) implementation for dimensionality reduction of the dataset's hidden states.
    """

    def __init__(self, n_components):
        super().__init__(n_components)
        self.reduced_dataset = None

    def get_hidden_states_reduction(self, hidden_states: dict):
        """
        Perform PCA dimensionality reduction on the dataset.

        :return: Reduced dataset's activations as a dictionary of pandas DataFrame (one for each layer).
        """
        reduced_hs_list = {}

        for hs_name in (x for x in hidden_states.column_names if x.startswith("hidden_state")):

            reduced_hs = TorchPCA(n_components=self.n_components).fit_transform(hidden_states[hs_name])
            reduced_hs_list[hs_name] = reduced_hs
            
        return reduced_hs_list
