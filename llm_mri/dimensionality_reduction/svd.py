from ._base import DimensionalityReduction
import torch

class SVD(DimensionalityReduction):
    """
    Principal Component Analysis (SVD) implementation for dimensionality reduction of the dataset's hidden states.
    """

    def __init__(self, n_components):
        super().__init__(n_components)
        self.reduced_dataset = None

    def get_reduction(self, hidden_states: dict):
        """
        Perform SVD dimensionality reduction on the dataset.

        :return: Reduced dataset's activations as a dictionary of pandas DataFrame (one for each layer).
        """
        reduced_hs_list = {}

        for hs_name in (x for x in hidden_states.column_names if x.startswith("hidden_state")):

            U, S, Vh = torch.linalg.svd(hidden_states[hs_name], full_matrices=False)

            # Keep only the top components
            reduced_hs = U[:, :self.n_components] @ torch.diag(S[:self.n_components])

            # Save result
            reduced_hs_list[hs_name] = reduced_hs

            
        return reduced_hs_list