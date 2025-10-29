from transformers import AutoTokenizer, AutoModel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout
import torch
from matplotlib.colors import Normalize
import numpy as np
from typing import Union, List
import datasets
from .graph import GraphND, Graph2D

class ActivationAreas:

    def __init__(self, model:str, dataset, reduction_method, device:str="cpu"):
        """
        Initializes the ActivationAreas class.

        Args:
            model (str): The model to be used.
            device (str): The device to be used (e.g., 'cpu' or 'cuda').
            dataset (Dataset): The dataset to be used.
        """
        self.model = model
        self.device = torch.device(device)
        self.dataset = dataset
        self.reduction_method = reduction_method
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.class_names = self.dataset.features['label'].names
        self.hidden_states_dataset = ""
        self.reduced_dataset = []
        self.num_layers = ""
        self.category_hidden_states = {}
        self.graph_class = ""


    def _tokenize(self, batch):
        """
        Tokenizes a batch of text.

        Args:
            batch (Dataset): Dataset with column "text" to be tokenized.

        Returns:
            Token: Tokenization of the Dataset, with padding enabled and a maximum length of 512.
        """
        if self.tokenizer.pad_token is None:  # Adding eos as pad token for decoders
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512)    

    def _initialize_dataset(self):
        """
        Initializes the encoded dataset from the model and transforms it into the torch type.

        Returns:
            Dataset: The encoded dataset in torch format.
        """
        dataset_encoded = self.dataset.map(
            self._tokenize, batched=True, batch_size=None)

        # Setting dataset to torch
        dataset_encoded.set_format("torch",
                                   columns=["input_ids", "attention_mask", "label"])
        return dataset_encoded

    def _extract_all_hidden_states(self, batch):
        """
        Extracts all hidden states for a batch of data.

        Args:
            batch (dict): Batch of data with model inputs.

        Returns:
            dict: Dictionary containing a tensor related to the extracted hidden layer weights and their respective labels.
        """
        
        model = AutoModel.from_pretrained(self.model).to(self.device)

        inputs = {k: v.to(self.device) for k, v in batch.items()
                  if k in self.tokenizer.model_input_names}

        with torch.no_grad():
            hidden_states = model(
                **inputs, output_hidden_states=True).hidden_states
        all_hidden_states = {}

        self.num_layers = len(hidden_states)
        
        for i, hs in enumerate(hidden_states):
            all_hidden_states[f"hidden_state_{i}"] = hs[:, 0].cpu().numpy()

        return all_hidden_states
    
    def process_activation_areas(self):
        """
        Processes the activation areas.

        Args:
            map_dimension (int): Size of the side of the square that will show the visualization.
        """

        # Obtaining the tokenized dataset
        self.dataset = self._initialize_dataset()

        # Extracting hidden states from the model
        self.hidden_states_dataset = self.dataset.map(self._extract_all_hidden_states, batched=True)

        # Reducing the hidden states dimensionality
        self.reduced_dataset = self.reduction_method.get_hidden_states_reduction(self.hidden_states_dataset)  

        # Definig the Graph object, based on the number of components
        if self.reduction_method.n_components == 2:
            
            self.graph_class = Graph2D(n_components=2,
                                        hidden_states=self.hidden_states_dataset,
                                        gridsize=self.reduction_method.gridsize,
                                        class_names=self.class_names,
                                        reduction_method=self.reduction_method,
                                        num_layers=self.num_layers)
        else:
            
            self.graph_class = GraphND(n_components=self.reduction_method.n_components,
                                        hidden_states=self.hidden_states_dataset,
                                        original_dataset=self.dataset,
                                        reduction_method=self.reduction_method,
                                        class_names=self.class_names,
                                        num_layers=self.num_layers)
            
    def get_grid(self, layer, category_name):
        
        if self.reduction_method.n_components != 2: # Grid cannot be obtained
            raise ValueError("Grid can only be obtained if the reduction method has 2 components. Please set the number of dimensions to 2.")
        
        return self.graph_class.get_grid(layer, category_name)
    
    def get_graph(self, categories: Union[str, List[str]], threshold: float = 0.3, gridsize: int = 10):
        """
        Temporary method to test the Graph class
        """
            
        g = self.graph_class.build_graph(categories, threshold)
        
        return g

    def get_graph_image(self, G: nx.Graph, colormap : str = 'coolwarm', fix_node_positions:bool = True):

        return self.graph_class.get_graph_image(G, colormap=colormap, fix_node_positions=fix_node_positions)

    def _get_nrag_embeddings(self):
        """
        Returns a dataset containing the reduced hidden states outputs for each category and another containing the labels.
        Both are going to be used on the train of a classifier.

        :return: Reduced hidden states as a pandas DataFrame and the labels of the dataset.
        """

        # Sort keys numerically to preserve layer order
        keys = sorted(self.reduced_dataset.keys(), key=lambda k: int(k.split('_')[-1]))

        # Concatenate tensors horizontally (to this phase, we consider all reduced layers components as features)
        concatenated = torch.cat([self.reduced_dataset[k] for k in keys], dim=1)

        # Building column names
        num_layers = len(keys)
        embedding_dim = concatenated.shape[1] // num_layers
        columns = [f"{layer+1}_{feat}" for layer in range(num_layers) for feat in range(embedding_dim)]

        # Convert to DataFrame
        nrag_embeddings = pd.DataFrame(concatenated.numpy(), columns=columns)
        labels = pd.DataFrame(self.dataset['label'])
        
        return nrag_embeddings, labels

            
    def _get_embeddings(self):
        """
        Returns the values on the last hidden state of the model (embeddings)
        """

        return pd.DataFrame(self.hidden_states_dataset[f'hidden_state_{self.num_layers-1}']), pd.DataFrame(self.dataset['label'])