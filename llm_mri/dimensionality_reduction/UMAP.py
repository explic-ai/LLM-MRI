from ._base import DimensionalityReduction
from umap import UMAP as UMAPLibrary
import torch
import numpy as np

class UMAP(DimensionalityReduction):
    """
    
    """

    def __init__(self, n_components, random_state=42, metric:str = "cosine"):
        super().__init__(n_components)
        self.reduced_dataset = None
        self.random_state = random_state
        self.metric = "cosine"
        
        
    def get_hidden_states_reduction(self, hidden_states: dict):
        """
        Perform UMAP dimensionality reduction on the dataset.

        :return: dict[str, torch.Tensor] shaped (n_samples, n_components)
        """
        reduced_hs_list = {}

        for hs_name in (x for x in hidden_states.column_names if x.startswith("hidden_state")):
            x = hidden_states[hs_name]  # expect torch.Tensor or array-like

            # Ensure NumPy for UMAP
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
                dtype, device = x.dtype, x.device
            else:
                x_np = np.asarray(x)
                dtype, device = torch.float32, "cpu"

            # UMAP fit/transform
            embedding_np = UMAPLibrary(n_components=self.n_components, metric=self.metric).fit_transform(x_np)

            # Convert back to torch (keep dtype/device if input was torch)
            embedding = torch.from_numpy(embedding_np).to(dtype=dtype, device=device)

            reduced_hs_list[hs_name] = embedding

        return reduced_hs_list
