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

    def get_reduction(self, hidden_states: dict):
        """
        Perform PCA dimensionality reduction on the dataset.

        :return: Reduced dataset's activations as a dictionary of pandas DataFrame (one for each layer).
        """
        reduced_hs_list = {}

        for hs_name in (x for x in hidden_states.column_names if x.startswith("hidden_state")):

            reduced_hs = TorchPCA(n_components=self.n_components).fit_transform(hidden_states[hs_name])
            reduced_hs_list[hs_name] = reduced_hs
            
        return reduced_hs_list
    
    def get_reduced_embeddings(self, hidden_states: dict, n_components: int = 0, n_layers: int = 0):
        """
        Obtain the reduced embeddings from the dataset's hidden states.

        :param hidden_states: The input dataset's hidden states as a dictionary of pandas DataFrame.
        :param n_components: The number of components to reduce to. If set to 0, returns the original embeddings.
        :return: Reduced embeddings as a dictionary of pandas DataFrame (one for each layer).
        """

        # If the number of components is set to 0, return the original embeddings reduced to n_components.
        # Else, concatenate columns from the hidden states. hidden states are in a dictionary, where each hs is a tensor and has its value on dataset['hidden_state_x'], for x in [1, 2, ..., n_layers].
        # So, concatenate all hidden states in the same dataset, and then get the reduction to n_components components.
        
        if n_components == 0:
            # Reduces the last embedding only to n_components
            return TorchPCA(n_components=n_components).fit_transform(hidden_states[f'hidden_state_{n_layers-1}'])

        # Concatenate all hidden states in one, and then reduce their dimensionality
        hs = [pd.DataFrame(hidden_states[f'hidden_state_{i}'].detach().cpu().numpy()) for i in range(n_layers)]
        concatenated_hs = pd.concat(hs, axis=1)

        reduced_hs = TorchPCA(n_components=n_components).fit_transform(concatenated_hs)
        
        # Convert the reduced hidden state to a DataFrame
        reduced_hs_df = pd.DataFrame(reduced_hs)
        return reduced_hs_df