from ._base import Graph
import torch
import networkx as nx
import pandas as pd
from typing import Union, List
from networkx.drawing.nx_agraph import graphviz_layout

class GraphND(Graph):

    def __init__(self, n_components: int, 
                 hidden_states: dict, 
                 original_dataset,
                 reduction_method, 
                 class_names, 
                 num_layers: int):

        """
        Initialize the GraphND object.

        :param n_components: The number of components that the graph was reduced based on.
        """
        super().__init__(n_components, hidden_states, reduction_method, class_names, num_layers)
        self.original_dataset = original_dataset
        self.category_hidden_states = {}

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
    
    def build_graph(self, category_list: Union[str, List[str]]=None, threshold: float=0.3) -> nx.Graph:
        """
        Builds the networkx graph for the network region activations,
        for a given label (category) passed as a parameter.

        Args:
            category_name (str): The name of the category. Default is an empty string.

        Returns:
            Graph: The networkx graph representing the activations.
        """
        
        if isinstance(category_list, str):
            category_list = [category_list]
        
        if len(category_list) > 2:
            raise ValueError("This method can only generate a graph for one or two categories. If you wish to compare more categories, please input one at a time")

        # List to store graphs for each category
        graphs_list = []

        for category in category_list:

            if category not in self.class_names:
                raise ValueError(f"Category '{category}' is not in the dataset's class names.")
            
            # Get the category index    
            category_index = self.class_names.index(category)    

            # Filter the dataset to get the indices of rows with the given category
            indices = [i for i, label in enumerate(self.original_dataset['label']) if label == category_index]
            
            #  Extract the rows from the hidden_states_dataset tensor
            filtered_hidden_states = self.hidden_states.select(indices) 
        
            #  Select only rows with selected categories from hidden state
            full_svd_hs = self.reduction_method.get_reduction(filtered_hidden_states)

            # 2) Select specific hidden states to compute spearman correlation
            c_hidden_states = {}

            for i in range(len(full_svd_hs)):
                c_hidden_states[f'hidden_state_{i}'] = full_svd_hs[f'hidden_state_{i}']
            
            # Updating graphs list
            graph = self._get_spearman_graph(c_hidden_states, category_index, threshold)
            graphs_list.append(graph)
            
            # Updating the hidden states for the given category
            self.category_hidden_states[category] = c_hidden_states

        # Merging graphs (if more than one category)
        if len(graphs_list) > 1:
            G = nx.compose(graphs_list[0], graphs_list[1])
            G.graph['label_names'] = [graphs_list[0].graph['label_names'][0], graphs_list[1].graph['label_names'][0]]
        
        else:
            G = graphs_list[0]

        # Defining the number of layers on the graph's properties
        G.graph['layers'] = self.num_layers

        return G

    def _get_node_positions(self, G, fix):
        """
        Contains the logic to generate the node positions for the graph.
        Args:
            G (nx.Graph): The graph for which to generate node positions.
            fix (bool): Whether to fix node positions based on their horizontal position.
            A fixed position enables comparison between different categories,
            while not fixing allows a better visualization of the graph.
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
            if fix == False:
                new_pos[node] = (pos[node][0], height_index)
                
            else:
                new_pos[node] = (width_index, height_index)

        return new_pos