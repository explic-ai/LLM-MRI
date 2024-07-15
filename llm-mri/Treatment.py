import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch #
from umap import UMAP #
from sklearn.preprocessing import MinMaxScaler #
import matplotlib.pyplot as plt
import numpy as np #
import sys
import seaborn as sns

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

    def set_device(self, device):
        """
        Sets the device that will be used by the class.

        Args:
            device (str): The device to be used (e.g., 'cpu' or 'cuda').
        """

        self.device = device
    
    def tokenize(self, batch):
        """
        Tokenizes a batch of text.

        Args:
            batch (Dataset): Dataset with column "text" to be tokenized.

        Returns:
            Token: Tokenization of the Dataset, with padding enabled and a maximum length of 512.
        """

        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


    def encode_dataset(self, dataset):
        """
        Maps over all items in the Dataset and performs tokenization.

        Args:
            dataset (Dataset): Dataset with text to be tokenized.

        Returns:
            Token: Tokenization of the Dataset, with padding enabled and a maximum length of 512.
        """

        dataset_encoded = dataset.map(self.tokenize, batched=True, batch_size=None)
        return dataset_encoded


    def set_embeddings_on_model(self, model_ckpt):
        """
        Sets embeddings on the model.

        Args:
            model_ckpt (Model): Hugging Face model.

        Returns:
            Model: Model passed as a parameter.
        """
        model = AutoModel.from_pretrained(model_ckpt).to(self.device)
        return model


    def extract_all_hidden_states(self, batch):
        """
        Extracts all hidden states for a batch of data.

        Args:
            batch (dict): Batch of data with model inputs.

        Returns:
            dict: Dictionary containing a tensor related to the extracted hidden layer weights and their respective labels.
        """
        
        model = self.set_embeddings_on_model(model_ckpt=self.model)

        inputs = {k:v.to(self.device) for k,v in batch.items() 
                if k in self.tokenizer.model_input_names}
        
        with torch.no_grad():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        all_hidden_states = {}
        
        for i, hs in enumerate(hidden_states):
            all_hidden_states[f"hidden_state_{i}"] = hs[:,0].cpu().numpy()
        
        return all_hidden_states


    def set_dataset_to_torch(self, dataset_encoded):
        """
        Sets the dataset format to PyTorch.

        Args:
            dataset_encoded (Dataset): Tokenized Dataset.

        Returns:
            Dataset: Dataset formatted for PyTorch.
        """

        dataset_encoded.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])
        return dataset_encoded


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
        mapper = UMAP(n_components=2, metric="cosine", random_state=42, n_jobs=1).fit(X_scaled) #, random_state=42
        df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
        df_emb["label"] = y
        return df_emb


    def get_grid(self, dataset, hidden_state_label, gridsize):
        """
        Returns a dataframe containing the embeddings for a specific layer of the network.

        Args:
            dataset (Dataset): The dataset to be used for the analysis.
            hidden_state_label (str): The hidden state label, such as 'hidden_state_name'.
            gridsize (int): The grid size.

        Returns:
            DataFrame: DataFrame containing the embeddings.
        """

        hidden_state = hidden_state_label.split("_")[-1]
        X = np.array(dataset[hidden_state_label])
        y = np.array(dataset["label"])
        df_emb = self.get_embeddings(X, y)
        df_emb = df_emb.assign(
        X=pd.cut(df_emb.X, gridsize, labels=False),
        Y=pd.cut(df_emb.Y, gridsize, labels=False)
        )
        # create a new column with the concatenation of X and Y

        df_emb['cell_label'] = hidden_state + "_" + df_emb['X'].astype(str) + "_" + df_emb['Y'].astype(str)
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
        
        #change figure title to hs
        full_name = f"{hidden_layer_name} : {label_name}"
        plt.title(full_name)
        
        return fig

    def get_all_grids(self, dataset, gridsize, buffer):
        
        for hs in [x for x in dataset.column_names if x.startswith("hidden_state")]:
            df_grid = self.get_grid(dataset, hs, gridsize)
            buffer.append(df_grid) # ith grid
        
        return buffer


sys.modules['Treatment'] = Treatment