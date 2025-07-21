import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from progress.bar import Bar
import networkx as nx


class Treatment:

    def __init__(self, model, device):
        """
        Initializes the Treatment class.

        Args:
            model (str): The model to be used.
            device (str): The device to be used (e.g., 'cpu' or 'cuda').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = device
        self.embeddings_dataset = []

    def get_embeddings_dataset(self):
        """
        Returns the embeddings dataset
        """
        return self.embeddings_dataset
 


    def spearman_correlation(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Spearman rank‐correlation matrix between columns of X and Y.

        Args:
            X: Tensor of shape (n_samples, n_features_X)
            Y: Tensor of shape (n_samples, n_features_Y)

        Returns:
            corr: Tensor of shape (n_features_X, n_features_Y),
                where corr[i, j] is Spearman's rho between X[:, i] and Y[:, j].
        """
        # 1) rank each column: argsort twice gives ranks 0..n-1
        rx = X.argsort(dim=0).argsort(dim=0).float()
        ry = Y.argsort(dim=0).argsort(dim=0).float()

        # 2) zero-mean
        rx -= rx.mean(dim=0, keepdim=True)
        ry -= ry.mean(dim=0, keepdim=True)

        # number of samples
        n = X.size(0)

        # 3) covariance of ranks (shape: n_features_X x n_features_Y)
        cov = (rx.t() @ ry) / (n - 1)

        # 4) standard deviations of ranks
        stdx = rx.std(dim=0, unbiased=True)    # shape (n_features_X,)
        stdy = ry.std(dim=0, unbiased=True)    # shape (n_features_Y,)

        # 5) outer product to normalize
        denom = stdx.unsqueeze(1) * stdy.unsqueeze(0)  # (n_features_X, n_features_Y)

        # 6) elementwise division → Spearman’s rho
        return cov / denom



    def get_embeddings(self, X, y):
        """
        Gets 2D embeddings by reducing the datasets dimensionalty for features and labels.

        Args:
            X (array): Matrix of embeddings for features.
            y (array): Embeddings vector for labels.

        Returns:
            DataFrame: DataFrame with 2D embeddings.
        """
        X_scaled = MinMaxScaler().fit_transform(X)

        mapper = UMAP(n_components=2, metric="cosine").fit(
            X_scaled)  # , random_state=42
        df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
        df_emb["label"] = y
        return df_emb

    def get_grid(self, dataset, hidden_state_label, gridsize):
        """
        Gets a dataframe containing the embeddings for a specific layer of the network, and
        cuts it to make the grid representation.

        Args:
            dataset (Dataset): The dataset to be used for the analysis.
            hidden_state_label (str): The hidden state label, such as 'hidden_state_name'.
            gridsize (int): The grid size.

        Returns:
            DataFrame: DataFrame containing the grid embeddings.
        """
        # Defining HS Label
        hidden_state = hidden_state_label.split("_")[-1]

        # Creating Documents and Target to be obtained the embeddings from
        X = np.array(dataset[hidden_state_label])
        y = np.array(dataset["label"])
        df_emb = self.get_embeddings(X, y)

        # Saving the embeddings of those documents
        self.embeddings_dataset.append(df_emb)

        # Turning into gridsize x gridsize
        df_emb = df_emb.assign(
            X=pd.cut(df_emb.X, gridsize, labels=False),
            Y=pd.cut(df_emb.Y, gridsize, labels=False)
        )

        # Adjusting Labels
        df_emb['cell_label'] = hidden_state + "_" + \
            df_emb['X'].astype(str) + "_" + df_emb['Y'].astype(str)
        return df_emb

    def get_activations_grid(self, hidden_layer_name, label, label_name, df_grid):
        """
        Reduces dimensionality and returns a NxN gridsize, each representing an activation region.

        Args:
            dataset (Dataset): The dataset to be used.
            gridsize (int): The grid size.
            hidden_layer_name (str): The hidden layer name, as 'hidden_layer_2'.
            label (int): The label as an integer.
            label_name (str): The label name.

        Returns:
            Figure: The activation grid plot for the specified layer and category.
        """

        df_grid = df_grid.loc[df_grid['label'] == label]

        ct = pd.crosstab(df_grid.Y, df_grid.X, normalize=False)

        ct = ct.sort_index(ascending=False)

        fig = sns.heatmap(ct, cmap="Blues", cbar=False, annot=True, fmt="d")

        # change figure title to hs
        full_name = f"{hidden_layer_name} : {label_name}"
        plt.title(full_name)

        return fig

    def get_all_grids(self, dataset, gridsize, buffer):
        """
        Gets and stores all dimension-reduced grids on buffer 

        Args:
            dataset (Dataset): The dataset to be used.
            gridsize (int): The grid size.
            buffer (list): Buffer to store grids

        Returns:
            buffer
        """
        with Bar('Processing layers...', max=len([x for x in dataset.column_names if x.startswith("hidden_state")])) as bar:
            for hs in [x for x in dataset.column_names if x.startswith("hidden_state")]:
                df_grid = self.get_grid(dataset, hs, gridsize)
                bar.next()
                buffer.append(df_grid)  # ith grid

        return buffer
