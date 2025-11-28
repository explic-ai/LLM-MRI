from .base import DimensionalityReduction
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional

class SVD(DimensionalityReduction):
    """
    Principal Component Analysis (SVD) implementation for dimensionality reduction of the dataset's hidden states.
    """

    def __init__(self, 
                 n_components: int,
                 random_state: Optional[int] = None,
                 gridsize: Optional[int] = 10):
        super().__init__(n_components)
        self.reduced_dataset = None
        self.gridsize = gridsize
        self.random_state = random_state
    def get_hidden_states_reduction(self, hidden_states: dict):
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
    

    def get_reduction(self, dataset):
        """
        SVD-based 2D reduction for a single dataset.
        Returns a DataFrame with columns ['X', 'Y'], to be used on the grids generation.
        """

        X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(dataset)

        # SVD on CPU tensor
        X_t = torch.from_numpy(np.asarray(X_std))
        U, S, Vh = torch.linalg.svd(X_t, full_matrices=False)

        # 2D coords
        coords = (U[:, :self.n_components] * S[:self.n_components]).cpu().numpy()
        return coords