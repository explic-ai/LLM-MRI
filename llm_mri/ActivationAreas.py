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
        self.reduced_dataset = self.reduction_method.get_reduction(self.hidden_states_dataset)
        

    def _spearman_correlation(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

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
    
    def _get_spearman_graph(self, reduced_hs_list, category_index, threshold):
        """
        Returns the networkx graph to represent the activations, using the Spearman correlation

        Args:
            reduced_hs_list (list): List of reduced hidden states.
            dim (int): The number of dimensions to reduce the activations to (default 40).
        
        Returns:
            Graph: The networkx graph representing the activations.
        """
        # Creating the graph
        G = nx.Graph()

        # Variable to store the correlation matrices
        correlation_reduced_hs = []

        # 2) Calculating correlation for every hidden state intersection
        for index in range(len(reduced_hs_list) - 1):
            first_layer = reduced_hs_list[f'hidden_state_{index}']
            second_layer = reduced_hs_list[f'hidden_state_{index+1}']

            correlation_matrix = self._spearman_correlation(
                first_layer, second_layer)
            
            # Generating names for columns and rows (hs{x}_{index})
            column_names = [f'{index}_{x}' for x in range(first_layer.shape[1])]
            row_names = [f'{index+1}_{x}' for x in range(first_layer.shape[1])] # Number of components

            # Disclaimer: The comparison is made between the components of the reduced dataset

            # Adding all different nodes to the graph
            G.add_nodes_from(column_names)
            G.add_nodes_from(row_names)

            # Turning matrix into DataFrame, so that components can be named
            spearman_matrix_df = pd.DataFrame(
                correlation_matrix.detach().numpy(), columns=column_names, index=row_names)

            # Storing matrix
            correlation_reduced_hs.append(spearman_matrix_df)

        # 3) Adding edges to the graph
        for corr_matrix in correlation_reduced_hs:
            for row_name, row_data in corr_matrix.iterrows():  # Iterating though rows
                for col_name, weight in row_data.items():  # Iterating through columns
                    if weight > threshold: # Threshold
                        # Adding edges
                        G.add_edge(col_name, row_name,
                                   weight=weight, label=category_index)
                    
                    # TODO: Add percentile as a parameter

        # Setting label names previously defined
        G.graph['label_names'] = self.class_names[category_index]

        if isinstance(G.graph['label_names'], str):
            G.graph['label_names'] = [self.class_names[category_index]]

        # Returning the full graph developed
        return G

    def get_graph(self, categories: Union[str, List[str]], threshold: float = 0.3):
        """
        Returns the networkx graph to represent the activations of one or more categories.

        Args:
            categories (list): list of strings, each representing a category from the dataset. 
            The method can generate a graph for one or two categories, but not more than that.
            threshold (float): The threshold of the spearman correlation between components (default 0.3).
            Edhes with correlation below the threshold will not be displayed.

        Returns:
            Graph: The networkx graph representing the activations.
        """

        if isinstance(categories, str):
            categories = [categories]
        
        if len(categories) > 2:
            raise ValueError("This method can only generate a graph for one or two categories. If you wish to compare more categories, please input one at a time")

        # List to store graphs for each category
        graphs_list = []

        for category in categories:

            if category not in self.class_names:
                raise ValueError(f"Category '{category}' is not in the dataset's class names.")
            
            # Get the category index    
            category_index = self.class_names.index(category)    

            # Filter the dataset to get the indices of rows with the given category
            indices = [i for i, label in enumerate(self.dataset['label']) if label == category_index]
            
            #  Extract the rows from the hidden_states_dataset tensor
            filtered_hidden_states = self.hidden_states_dataset.select(indices) 
        
            #  Select only rows with selected categories from hidden state
            full_svd_hs = self.reduction_method.get_reduction(filtered_hidden_states)

            # 2) Select specific hidden states to compute spearman correlation
            c_hidden_states = {}

            for i in range(len(full_svd_hs)):
                c_hidden_states[f'hidden_state_{i}'] = full_svd_hs[f'hidden_state_{i}']
            
            # Updating graphs list
            graphs_list.append(self._get_spearman_graph(c_hidden_states, category_index, threshold))
        
        # Merging graphs (if more than one category)
        if len(graphs_list) > 1:
            G = nx.compose(graphs_list[0], graphs_list[1])
            G.graph['label_names'] = [graphs_list[0].graph['label_names'][0], graphs_list[1].graph['label_names'][0]]
        
        else:
            G = graphs_list[0]

        # Defining the number of layers on the graph's properties
        G.graph['layers'] = self.num_layers

        return G
        

  
    def get_graph_image(self, G: nx.Graph, colormap : str = 'coolwarm', fix_node_dimensions:bool = True):
        """
        Generates a matplotlib figure of the graph with nodes as pizza graphics.
        Args:
        G (networkx.Graph): The NetworkX graph.
        colormap (string): A string referent to the desired colormap. default is set by 'bwr'.
        fix_node_dimensions (bool): If True, the horizontal position of the node determines the dimension being represented by the node.
        If False, the horizontal position is defined by the layout algorithm.
        
        Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure representing the graph.
        """

        # Verifying if graph passed is a networkx graph
        if not isinstance(G, nx.Graph):
            raise TypeError("The graph must be a networkx Graph object.")
        
        # Verifying if the graph has nodes
        if G.number_of_nodes() == 0:
            raise ValueError("The graph has no nodes to display.")
        
        # Verifying if the graph has edges
        if G.number_of_edges() == 0:
            raise ValueError("The graph has no edges to display.")
        
        # Get all nodes from the defined category(ies) graph
        nodelist = list(G.nodes())

        # Use graphviz_layout for positioning
        pos = graphviz_layout(G, prog="dot")

        # Fixing node positions        
        new_pos = {}

        for node in nodelist:
            
            # Extract the first character to determine height index
            height_index = int(node.split('_')[0])
            width_index = int(node.split('_')[-1])

            # If fix_node_dimensions is True, the horizontal position determines the dimension being represented by the node.
            if fix_node_dimensions == False:
                new_pos[node] = (pos[node][0], height_index)
                
            else:
                new_pos[node] = (width_index, height_index)
            
        pos = new_pos
        
        # Create the matplotlib figure
        fig, ax = plt.subplots(figsize=(25, 6))
        
        # Generate edge colors
        edge_colors = self._generate_graph_edge_colors(G, colormap)

        ordered_edge_colors = edge_colors

        # Create a mapping from label to color
        if len(edge_colors) > 1:
            
            # Define your custom color mapping for labels
            custom_colors = {
                0: edge_colors[0], # Color of first category 
                1: edge_colors[1], # Color of second category
            }

            # Generate edge_colors list aligned with the edgelist
            ordered_edge_colors = []

            # Getting the unique label's indixes
            unique_labels = [self.class_names.index(categ) for categ in G.graph['label_names']]

            for u, v in G.edges().keys():

                # Get the index of the label for the edge
                index = unique_labels.index(G[u][v].get('label', 0))

                color = custom_colors.get(index, 'gray') 

                # Getting the color for every edge
                ordered_edge_colors.append(color)
        
        # Coloring Nodes
        node_colors = self._generate_node_colors(G, colormap)
        
        # Create legend handles based on edge colors
        legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in (edge_colors)]
        plt.legend(legend_handles, G.graph['label_names'], loc='upper right')
        
        # Compute the degree of each node
        degrees = dict(G.degree())
        
        # Scale node sizes
        max_degree = max(max(degrees.values()), 4)
        node_sizes = [100 + (degrees[node] / max_degree) * 1400 for node in nodelist]
        
        # Draw edges with specified widths and colors
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=G.edges(),
            width=[edge[-1]['weight'] * 1.5 for edge in G.edges(data=True)],
            edge_color=ordered_edge_colors,
            alpha=0.9,
            ax=ax
        )
        
        # Draw nodes with sizes proportional to their degree
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=node_sizes,
            node_color=node_colors, # added
            alpha=0.9,
            linewidths=1,
            edgecolors='black'
        )
        
        # Remove axes for a cleaner look
        plt.axis('off')
        
        return fig
    
    def _generate_node_colors(self, G, colormap: str = 'coolwarm'):

        """
        Generates a list of colors based on the amount of nodes in the graph's edges, being
        each node color's proportional to the amount of times the node was activated by a label.

        Args:
            G (Graph): The networkx graph.
            edge_colors (list): List containing the edge colors

        Returns:
            list: A list of colors for the graph's nodes.
        """

        if len(G.graph['label_names']) < 2:
            # single feature being analyzed
            return ['gray']
        
        # category_index = self.class_names.index(category)    

        # assign labels to variables for clarity
        label1, label2 = [self.class_names.index(categ) for categ in G.graph['label_names']] # as previously defined

        # initialize dictionaries to count label activations per node
        label1_counts = {node: 0 for node in G.nodes()}
        label2_counts = {node: 0 for node in G.nodes()}

        # iterate over all edges to count label activations
        for u, v, data in G.edges(data=True):
            label = data.get('label')
            weight = data.get('weight', 1)  # default weight is 1 if not specified

            if label == label1:
                label1_counts[u] += weight
                label1_counts[v] += weight

            elif label == label2:
                label2_counts[u] += weight
                label2_counts[v] += weight
            
            else: # label == 2

                label1_counts[u] += weight
                label1_counts[v] += weight
                label2_counts[u] += weight
                label2_counts[v] += weight

        # retrieve the specified divergent colormap
        cmap = plt.get_cmap(colormap)

        # initialize Normalize object to map ratios between 0 and 1
        norm = Normalize(vmin=0, vmax=1)

        # compute proportions and assign node colors
        node_colors = []
        for node in G.nodes():
            count_label1 = label1_counts.get(node, 0)
            count_label2 = label2_counts.get(node, 0)
            total = count_label1 + count_label2

            if total > 0:
                ratio = count_label2 / total  # proportion of label2
            else:
                ratio = 0.5  # neutral ratio if no connected edges

            # normalize the ratio
            norm_ratio = norm(ratio)

            # map the normalized ratio to a color using the colormap
            color = cmap(norm_ratio)

            # convert rgb to hex
            color_hex = mcolors.to_hex(color)
            node_colors.append(color_hex)

        return node_colors

    def _generate_graph_edge_colors(self, G, colormap='coolwarm'):
            """
            Generates a list of colors based on the number of labels in the graph's edges.

            Args:
                G (Graph): The networkx graph.
                colormap (str): The name of the Matplotlib colormap to use (default 'bwr').

            Returns:
                list: A list of HEX color codes for the graph's edges.
            """
            # extract edge attributes from the first edge
            first_edge_attrs = list(G.edges(data=True))
        
            edge_attributes = list(first_edge_attrs[0][-1].keys())

            if 'label' in edge_attributes:

                # extract all labels from the edges
                unique_labels = [self.class_names.index(categ) for categ in G.graph['label_names']] # as previously defined

                num_labels = min(2, len(unique_labels))  # handles up to two labels

                # retrieve the specified continuous colormap
                colormap_list = plt.get_cmap(colormap)

                # generate evenly spaced values between 0 and 1 for sampling the colormap
                color_values = np.linspace(0, 1, num_labels)

                # sample the colormap
                colors = [(colormap_list(value)) for value in color_values]

                return colors
            
            else:
                return ['lightblue']
        
    def get_nrag_embeddings(self, n_components:int=None):
        """
        This method returns either the reduced version of all hidden states combined, or only the embedding, last hidden state of the model.
        It also returns the label of each instance, so that it can be used to train a classifier

        :param n_components: The number of components to reduce to. If set to None, returns the original embeddings.
        :return: Reduced embeddings as a dictionary of pandas DataFrame (one for each layer) and the labels of the dataset.
        """

        print(self.dataset['label'].shape)
        # If some reduction should be made, call the reduction method
        if n_components is not None:
            return self.reduction_method.get_reduced_embeddings(self.hidden_states_dataset, n_components, self.num_layers), pd.DataFrame(self.dataset['label'])
        
        # Else, return the original embeddings
        else:
            return pd.DataFrame(self.hidden_states_dataset[f'hidden_state_{self.num_layers-1}']), pd.DataFrame(self.dataset['label'])
        