from abc import ABC, abstractmethod
from typing import List, Union, Optional
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import numpy as np

class Graph(ABC):
    """
    Abstract base class for building the graph accordingly to the dimensionality reduction type and number of components.
    """

    def __init__(self, n_components: int, hidden_states: dict, reduction_method, class_names, num_layers):
        """
        Initialize the graph object.

        :param n_components: The number of components that the graph was reduced based on.
        """
        self.n_components = n_components
        self.hidden_states = hidden_states
        self.reduction_method = reduction_method
        self.class_names = class_names
        self.num_layers = num_layers

    @abstractmethod
    def build_graph(self, category_list: Union[str, List[str]]=None, threshold: Optional[float]=0.3):

        """

        """
        pass   
    
    @abstractmethod
    def _get_node_positions(self, graph: nx.graph, fix: bool):
        """
        Contains the logic to generate the node positions for the graph.
        Args:
            G (nx.Graph): The graph for which to generate node positions.
            fix (bool): Whether to fix node positions based on their horizontal position.
                        A fixed position enables comparison between different categories,
                        while not fixing allows a better visualization of the graph.
        """
        pass

    def _generate_node_colors(self, G, colormap:str = 'coolwarm'):

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
        
        # assign labels to variables for clarity
        label1, label2 = [self.class_names.index(categ) for categ in G.graph['label_names']]

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

    def _generate_graph_edge_colors(self, G, colormap:str = 'coolwarm'):
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
    
    def get_graph_image(self, G: nx.Graph, colormap:str = 'coolwarm', fix_node_dimensions:bool = True, fix_node_positions:bool = True) -> plt.Figure: 
        """
        Renders generated graph as an image
        Args:
        G (networkx.Graph): The NetworkX graph.
        colormap (string): A string referent to the desired colormap. default is set by 'bwr'.
        fix_node_dimensions (bool): If True, the horizontal position of the node determines the dimension being represented by the node.
        If False, the horizontal position is defined by the layout algorithm.
        
        Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure representing the graph.
        """

        # Get all nodes from the defined category(ies) graph
        nodelist = list(G.nodes())

        if self.reduction_method.n_components == 2:
            # Renders visualization for the 2d Graph
            pos = self._get_node_positions(G, fix_node_positions)
        
        else:
            # Renders visualization for the ND Graph
            pos = self._get_node_positions(G, fix_node_dimensions)

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
            node_color=node_colors, 
            alpha=0.9,
            linewidths=1,
            edgecolors='black'
        )
        
        # Optionally, add labels to nodes (commented out for clarity, but could be parametrized later if needed)

        # node_labels = {n: str(n) for n in nodelist}

        # nx.draw_networkx_labels(
        #     G,
        #     pos,
        #     labels=node_labels,
        #     font_size=10,
        #     font_color="black",
        #     ax=ax
        # )
        
        # Remove axes for a cleaner look
        plt.axis('off')
        
        return fig
   