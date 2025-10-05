from ._base import DimensionalityReduction
from torchdr import PCA as TorchPCA
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

    def get_reduction(self, dataset):
        """
        PCA-based 2D reduction for a single dataset.
        Returns a DataFrame with columns ['X', 'Y'], to be used on the grids generation.
        """

        # 1) Scale like your UMAP version
        X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(dataset)

        # 2) 2D PCA (using TorchPCA to stay consistent with your class)
        coords = TorchPCA(n_components=2).fit_transform(X_std)

        # 3) Return as a DataFrame
        df_emb = pd.DataFrame(coords, columns=["X", "Y"])
        return df_emb
