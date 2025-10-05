from .base import DimensionalityReduction
from torchdr import PCA as TorchPCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional

class PCA(DimensionalityReduction):
    """
    Principal Component Analysis (PCA) implementation for dimensionality reduction of the dataset's hidden states.
    """

    def __init__(self, n_components: int, 
                 gridsize: Optional[int] = 10,
                random_state: Optional[int] = None):
        super().__init__(n_components, random_state)
        self.reduced_dataset = None
        self.random_state = random_state
        self.gridsize = gridsize

    def get_hidden_states_reduction(self, hidden_states: dict):
        """
        Perform PCA dimensionality reduction on the dataset.

        :return: Reduced dataset's activations as a dictionary of pandas DataFrame (one for each layer).
        """
        reduced_hs_list = {}

        for hs_name in (x for x in hidden_states.column_names if x.startswith("hidden_state")):
            
            if self.random_state is not None:
                reduced_hs = TorchPCA(n_components=self.n_components, random_state=self.random_state).fit_transform(hidden_states[hs_name])
            
            else:
                reduced_hs = TorchPCA(n_components=self.n_components).fit_transform(hidden_states[hs_name])
                
            reduced_hs_list[hs_name] = reduced_hs
            
        return reduced_hs_list

    def get_reduction(self, dataset):
        """
        PCA-based 2D reduction for a single dataset.
        Returns a DataFrame with columns ['X', 'Y'], to be used on the grids generation.
        """

        X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(dataset)

        if self.random_state is not None:
            coords = TorchPCA(n_components=2, random_state=42).fit_transform(X_std)
        
        else:
            coords = TorchPCA(n_components=2).fit_transform(X_std)


        df_emb = pd.DataFrame(coords, columns=["X", "Y"])
        return df_emb
