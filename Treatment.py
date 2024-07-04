import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch #
from umap import UMAP #
from sklearn.preprocessing import MinMaxScaler #
import matplotlib.pyplot as plt
import numpy as np #
import sys
from datasets import ClassLabel

class Treatment:

    def __init__(self, model, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = device

    def setModel(self, model):
        '''
        Sets the model stored by the class.
        '''

        self.model = model

    def setDevice(self, device):
        '''
        Sets the device that will be used by the class.
        '''

        self.device = device
    
    def setTokenizer(self, tokenizer):
        '''
        Sets the tokenizer that will be used by the class.
        '''

        self.tokenizer = tokenizer

    def tokenize(self, batch):
        '''
        Input: (Dataset) Dataset with text to be tokenized.
        Output: (Token) Tokenization of the Dataset, with padding enabled and a maximum length of 512.
        '''

        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


    def encodeDataset(self, dataset):
        '''
        Input: (Dataset) Dataset with text to be tokenized.
        Output: (Token) Tokenization of the Dataset, with padding enabled and a maximum length of 512.

        Maps over all items in the Dataset and performs tokenization.
        '''

        dataset_encoded = dataset.map(self.tokenize, batched=True, batch_size=None)
        return dataset_encoded


    def saveDataset(self, dataset):
        '''
        Input: (Dataset) Tokenized Dataset.

        Saves the Dataset to disk.
        '''

        dataset.save_to_disk("dataset_encoded.hf")


    def setEmbeddingsOnModel(self, model_ckpt):
        '''
        Input: (Model) Hugging Face model.
        Output: (Model) Model passed as a parameter.
        '''

        model = AutoModel.from_pretrained(model_ckpt).to(self.device)
        return model


    def extract_all_hidden_states(self, batch):
        '''
        Input: (dict) Batch of data with model inputs.
        Output: (dict) Dictionary containing a tensor related to the extracted hidden layer weights and their respective labels.

        This function extracts for all hidden layers of the model.
        '''

        model = self.setEmbeddingsOnModel(model_ckpt=self.model)

        inputs = {k:v.to(self.device) for k,v in batch.items() 
                if k in self.tokenizer.model_input_names}
        
        with torch.no_grad():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        all_hidden_states = {}
        
        for i, hs in enumerate(hidden_states):
            all_hidden_states[f"hidden_state_{i}"] = hs[:,0].cpu().numpy()
        
        return all_hidden_states


    def setDatasetToTorch(self, dataset_encoded):
        '''
        Input: (Dataset) Tokenized Dataset.
        Output: (Dataset) Dataset formatted for PyTorch.
        '''

        dataset_encoded.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])
        return dataset_encoded


    def get_embeddings(self, X, y):
        '''
        Input: (array) Matrix of embeddings for features X.
            (array) Embeddings vector for labels y.
        Output: (DataFrame) DataFrame with 2D embeddings.
        '''

        X_scaled = MinMaxScaler().fit_transform(X)
        mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled) #, random_state=42
        df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
        df_emb["label"] = y
        return df_emb


    def plot_map(self, dataset, hidden_state_label, map_dimension):
        '''
        Input: (Dataset) Dataset containing the hidden layers.
            (string) Hidden layer categories to be plotted.
        Output: Displays a 2D scatter plot of the embeddings.
        '''

        X = np.array(dataset[hidden_state_label])
        y = np.array(dataset["label"])
        df_emb = self.get_embeddings(X, y)
        fig, axes = plt.subplots(1, 2, figsize=(7,5)) # gotta add more in case there are more than two features
        axes = axes.flatten()
        cmaps = ["Blues", "Reds"]
        labels = dataset.features["label"].names

        for i, (label, cmap) in enumerate(zip(labels, cmaps)):
            df_emb_sub = df_emb.query(f"label == {i}")
            axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                        gridsize=map_dimension, linewidths=(0,)) # 
            axes[i].set_title(label)
            axes[i].set_xticks([]), axes[i].set_yticks([])

        fig.suptitle(hidden_state_label, fontsize=16)

        return fig
    

    def get_grid(self, dataset, hidden_state_label, gridsize):
        '''
        Returns a dataframe containing the embeddings for each hidden layer of the network.
        '''

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


sys.modules['Treatment'] = Treatment