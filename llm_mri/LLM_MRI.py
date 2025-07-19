from llm_mri.Treatment import Treatment
from llm_mri.dimensionality_reduction import PCA
from transformers import AutoTokenizer
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import matplotlib.cm as cm

class LLM_MRI:

    def __init__(self, model, device, dataset, reduction_method):
        """
        Initializes the LLM_MRI class.

        Args:
            model (str): The model to be used.
            device (str): The device to be used (e.g., 'cpu' or 'cuda').
            dataset (Dataset): The dataset to be used.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = torch.device(device)
        self.dataset = dataset
        self.base = Treatment(model, device)
        self.gridsize = 10
        self.class_names = self.dataset.features['label'].names
        self.dataset = self.initialize_dataset() 
        self.hidden_states_dataset = ""
        self.reduced_dataset = []
        self.label_names = []
        self.graph = ""
        self.svd_graph = ''
        self.current_category = 0
        self.dim = 2
        self.threshold = 0.3
        self.reduction_method = reduction_method

    def initialize_dataset(self):
        """
        Initializes the encoded dataset from the model and transforms it into the torch type.

        Returns:
            Dataset: The encoded dataset in torch format.
        """
        encodedDataset = self.base.encode_dataset(self.dataset)

        # Transformando o dataset para o formato Torch
        encodedDataset = self.base.set_dataset_to_torch(encodedDataset)
        return encodedDataset


    def process_activation_areas(self, map_dimension:int):
        """
        Processes the activation areas.

        Args:
            map_dimension (int): Size of the side of the square that will show the visualization.
        """
    
        datasetHiddenStates = self.dataset.map(self.base.extract_all_hidden_states, batched=True)

        self.gridsize = map_dimension
        self.hidden_states_dataset = datasetHiddenStates

        # Getting the grid dataset
        self.reduced_dataset = self.base.get_all_grids(datasetHiddenStates, map_dimension, self.reduced_dataset)
        
        # Performing the SVD for hidden states
        self.svd_graph = self.get_svd_graph()

        # Getting the original graph
        self.graph = self.get_graph() # To be altered


    def get_layer_image(self, layer:int, category:int):
        """
        Gets the activation grid for the desired layer and category.

        Args:
            layer (int): The layer to be visualized.
            category (int): The category whose activations will be visualized.

        Returns:
            figure (plt.figure): The activation grid plot for the specified layer and category.
        """

        # Obtaining layer name string 
        hidden_name = f"hidden_state_{layer}"

        category_to_int = self.class_names.index(category)
        return self.base.get_activations_grid(hidden_name, category_to_int, category, self.reduced_dataset[layer])
       

    def get_original_map(self, layer:int, colormap:str = 'viridis'):
        """
        Returns a scatterplot with the UMAP representation for every example on the corpus
        passed by the user for a specific layer.

        Args:
            layer (int): The layer to be visualized.
            colormap (int): The colormap to be used on the display.

        Returns:
            figure (plt.figure): Plot representing 2-dimension UMAP representation of documents
            for a specific layer before turning it into a grid visualization.
        """
        
        # Selecting reduced dataset (grids)
        data_points = self.base.get_embeddings_dataset()[layer]

        # Defining colormap
        num_categories = len(self.class_names)
        cmap = plt.get_cmap(colormap, num_categories)  # Get the colormap
        colors = [cmap(i) for i in range(num_categories)]

        # Create the figure and axes
        fig, ax = plt.subplots()
        
        # Iterate over unique categories
        for i, category in enumerate(data_points['label'].unique()):

            # Filter the DataFrame for the current category
            category_df = data_points[data_points['label'] == category]
            X = category_df['X']
            Y = category_df['Y']

            # Plot scatter points on the axes
            ax.scatter(X, Y, color=colors[i], label=self.class_names[category], alpha=0.9) 

        # Add legend and labels
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"UMAP representation for hidden state: {layer}")
        
        return ax


    def get_graph(self, category_name:str = ""):
        """
        Builds the pandas edgelist (graph representation) for the network region activations,
        for a given label (category) passed as a parameter.

        Args:
            category_name (str): The name of the category. Default is an empty string.

        Returns:
            Graph: The networkx graph representing the activations.
        """

        if(category_name != ""):
            category = self.class_names.index(category_name)
            self.label_names.append(category_name)

        else:
            category = -1
        
        # Obtaining desired layer
        datasetHiddenStates = self.hidden_states_dataset
        
        hss = [x for x in datasetHiddenStates.column_names if x.startswith("hidden_state")]

        # crete an empty dataframe with columns up, down and corr
        df_graph = pd.DataFrame(columns=["cell_label_1", "cell_label_2", "weight", "level"])

        for hs in range(0, len(hss)-1):
            
            df_grid1 = self.reduced_dataset[hs]
            df_grid2 = self.reduced_dataset[hs+1]

            # when no category is passed gets all values
            if category == -1:
                df_join = df_grid1[['cell_label']].join(df_grid2[['cell_label']], lsuffix='_1', rsuffix='_2')
            
            # when category is passed filters values by category
            else:
                df_join = df_grid1.loc[df_grid1['label'] == category][['cell_label']].join(df_grid2.loc[df_grid2['label'] == category][['cell_label']], lsuffix='_1', rsuffix='_2')

            # group by and count the number of rows
            df_join_grouped = df_join.groupby(['cell_label_1', 'cell_label_2']).size().reset_index(name='weight')

            df_join_grouped['level'] = hs

            df_graph = pd.concat([df_graph, df_join_grouped])

            
        G = nx.from_pandas_edgelist(df_graph, 'cell_label_1', 'cell_label_2', ['weight'])

        # when generating category graph, assigns category label to edges
        if category != -1:
            nx.set_edge_attributes(G, category, "label")

        # Setting dimensionality reduction type to UMAP
        G.graph['dimensionality_reduction'] = "UMAP"

        G.graph['label_names'] = self.label_names

        return G


    def get_spearman_graph(self, reduced_hs_list):
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

            correlation_matrix = self.base.spearman_correlation(
                first_layer, second_layer)
            
            # Generating names for columns and rows (hs{x}_{index})
            column_names = [f'{index}_{x}' for x in range(first_layer.shape[0])]
            row_names = [f'{index+1}_{x}' for x in range(first_layer.shape[0])] # Number of instances
            
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
                    if weight > self.threshold: # Threshold
                        # Adding edges
                        G.add_edge(col_name, row_name,
                                   weight=weight, label=self.current_category)
                    
                    # TODO: Add percentile as a parameter

        # Setting dimensionality reduction type to SVD
        G.graph['dimensionality_reduction'] = "SVD"

        # Setting label names previously defined
        G.graph['label_names'] = self.label_names

        # Returning the full graph developed
        return G


    def get_svd_reduction(self, dataset_hidden_states=None, dim:int=40, category:str = ""):
        """
        Builds the networkx graph to represent the activations, using the SVD dimensionality reduction.

        Args:
            dim (int): The number of dimensions to reduce the activations to (default 40).

        Returns:
            Graph: The networkx graph representing the activations.
        """

        reduced_hs_list = []

        if dataset_hidden_states is None:
            dataset_hidden_states = self.hidden_states_dataset
        
        # 1) Reducing dimensionality through PCA
        for hs_name in [x for x in dataset_hidden_states.column_names if x.startswith("hidden_state")]:

            reduced_hs = PCA(n_components=dim).fit_transform(dataset_hidden_states[hs_name])
            reduced_hs_list.append(reduced_hs)
            

        return reduced_hs_list


    def get_svd_graph(self):
        """
        Builds the networkx graph to represent the activations, using the SVD dimensionality reduction.

        Args:
            dim (int): The number of dimensions to reduce the activations to (default 40).

        """

        reduced_hidden_states = self.reduction_method.get_reduction(self.hidden_states_dataset)
        return self.get_spearman_graph(reduced_hidden_states)


    def get_composed_svd_graph(self, category1, category2, dim:int=16, threshold:float=0.3):
        """
        Returns a composed graph for two categories, using the SVD dimensionality reduction.

        Args:
            category1 (str): The first category from the passed documents to be displaced on the graph.
            category2 (str): The second category from the passed documents to be displayed on the graph.
            dim (int): The number of dimensions to reduce the activations to (default 16).
            threshold (float): The threshold of the spearman correlation between components (default 0.3). The greater the threshold, the fewer edges will be displayed. 
            Threshold of 0 means that every edge is being displayed, and the threshold of 1 means that no edge is being displayed.
        """
        self.threshold = threshold

        # 1) Generate graph of only requested labels
        
        # Get the category index
        category1_index = self.class_names.index(category1)
        category2_index = self.class_names.index(category2)

        # Filter the dataset to get the indices of rows with the given category
        indices = [i for i, label in enumerate(self.dataset['label']) if label == category1_index or label == category2_index]

        # Extract the rows from the hidden_states_dataset tensor
        filtered_hidden_states = self.hidden_states_dataset.select(indices) 
        
        # Select only rows with selected categories from hidden state
        full_svd_hs = self.reduction_method.get_reduction(filtered_hidden_states)

        # 2) Select specific hidden states to compute spearman correlation
        
        # The first indices are going to be equivalent to the first categories, the next ones to the second
        # To be Altered
        # indices_categ1 = list(range(int(len(full_svd_hs[0])/2)))
        # indices_categ2 = list(range(int(len(full_svd_hs[0])/2), int(len(full_svd_hs[0]))))

        indices_categ1 = [i for i, label in enumerate(filtered_hidden_states['label']) if label == category1_index]
        indices_categ2 = [i for i, label in enumerate(filtered_hidden_states['label']) if label == category2_index] 

        # Extract full hidden state list from indices
        c1_hidden_states = {}
        c2_hidden_states = {}

        for i in range(len(full_svd_hs)):
            c1_hidden_states[f'hidden_state_{i}'] = full_svd_hs[f'hidden_state_{i}'][indices_categ1]
            c2_hidden_states[f'hidden_state_{i}'] = full_svd_hs[f'hidden_state_{i}'][indices_categ2]

        # Generate graph for first category
        c1_graph = self.get_spearman_graph(c1_hidden_states)

        # Updating category
        self.current_category = 1

        # Generate graph for the second category
        c2_graph = self.get_spearman_graph(c2_hidden_states)

        # Reseting category
        self.current_category = 0

        # Since nodes are the same, full graph are going to contain the same nodes
        G_composed = nx.Graph()
        G_composed.add_nodes_from(c1_graph.nodes())
        
        # For the edges, we need to concatenate the edges from the two obtained graph
        G_composed.add_edges_from(c1_graph.edges(data=True))
        G_composed.add_edges_from(c2_graph.edges(data=True))

        # Defining dimensionality reduction attribute
        G_composed.graph['dimensionality_reduction'] = "SVD"

        # Adding Label names to assigned variable
        self.label_names.append(category1)
        self.label_names.append(category2)

        # Adding labels to graph property
        G_composed.graph['label_names'] = self.label_names

        self.svd_graph = G_composed
        
        return G_composed


    def generate_graph_edge_colors(self, G, colormap='coolwarm'):
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
            labels = [data['label'] for _, _, data in G.edges(data=True) if 'label' in data]
            unique_labels = sorted(set(labels))
            num_labels = min(2, len(unique_labels))  # handles up to two labels

            # retrieve the specified continuous colormap
            colormap_list = plt.get_cmap(colormap)

            # generate evenly spaced values between 0 and 1 for sampling the colormap
            color_values = np.linspace(0, 1, num_labels)

            # sample the colormap
            colors = [(colormap_list(value)) for value in color_values]

            if len(self.label_names) > 1:

                    blended_rgb = tuple(0.5 * c1 + 0.5 * c2 for c1, c2 in zip(colors[0], colors[1]))

                    colors.append(blended_rgb)

            return colors
        
        else:
            return ['lightblue']

        
    def get_graph_image(self, G, colormap = 'coolwarm', fix_node_positions:bool = True, fix_node_dimensions:bool = True):
        """
        Generates a matplotlib figure of the graph with nodes as pizza graphics.
        Args:
        G (networkx.Graph): The NetworkX graph.
        colormap (string): A string referent to the desired colormap. default is set by 'bwr'.

        Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure representing the graph.
        """

        # By default, full_graph is the current graph
        full_graph = G

        if fix_node_positions: # If asked to fix, the graph of all categories will be considered
            
            if G.graph['dimensionality_reduction'] == "SVD":
                full_graph = self.svd_graph

            else:
            # Getting the graph with all possible activations
                full_graph = self.graph

        # Get all nodes from the defined category(ies) graph
        nodelist = list(G.nodes())

        # Use graphviz_layout for positioning
        pos = graphviz_layout(full_graph, prog="dot")

        # Since pos was generated to all nodes, we are going to remove the ones that are not currently being displaced
        removed_nodes = []
        for node in pos.keys():
            if node not in nodelist:
                removed_nodes.append(node)
        
        # Removing nodes
        for node in removed_nodes:
            pos.pop(node)

        # Fixing node positions        
        new_pos = {}

        for node in nodelist:
            
            # Extract the first character to determine height index
            height_index = int(node.split('_')[0])  # Adjust based on your node naming convention
            width_index = int(node.split('_')[-1])

            if G.graph['dimensionality_reduction'] == "SVD":

                if fix_node_dimensions == False:
                    new_pos[node] = (pos[node][0], height_index)
                
                else:
                    new_pos[node] = (width_index, height_index)
            
            else: # Dimensionality reduction is UMAP
                new_pos[node] = (pos[node][0], height_index)

        pos = new_pos
        
        # Create the matplotlib figure
        fig, ax = plt.subplots(figsize=(25, 6))
        
        # Generate edge colors
        edge_colors = self.generate_graph_edge_colors(G, colormap)

        ordered_edge_colors = edge_colors

        # Create a mapping from label to color
        if len(edge_colors) > 2:

            # Define your custom color mapping for labels
            custom_colors = {
                0: edge_colors[0],  
                1: edge_colors[1], 
                2: edge_colors[2]  
            }

            # Generate edge_colors list aligned with the edgelist
            ordered_edge_colors = [
                custom_colors.get(G[u][v].get('label', 0), 'gray') 
                for u, v in G.edges().keys()]
        
        # Coloring Nodes
        node_colors = self.generate_node_colors(G, colormap)
        
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
            width=[edge[-1]['weight'] * 2 for edge in G.edges(data=True)],
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
        
        # Clear label names for future use
        self.label_names = []
        
        # Remove axes for a cleaner look
        plt.axis('off')
        
        return fig


    def get_composed_graph(self, category1, category2):
        """
        Displays the pandas edgelist (graph representation) for the network region activations,
        being each label represented by a different edge color.

        Args:
            category1 (String): The first category.
            category2 (String): The second category.

        Returns:
            fig (plt.figure): The matplotlib figure representation of the graph.
        """
        g1 = self.get_graph(category1)
        g2 = self.get_graph(category2)
        
        for u, v, data in g1.edges(data=True):
            data['label'] = 0
        
        for u, v, data in g2.edges(data=True):
            data['label'] = 1

        g_composed = nx.compose(g1, g2)

        # Marking repeated edges
        duplicates = list(set(g1.edges) & set(g2.edges))
        for e in duplicates : g_composed.edges[e]['label'] = 2 
        
        return g_composed

    
    def generate_node_colors(self, G, colormap: str = 'coolwarm'):

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
        label1, label2 = [0, 1] # as previously defined

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

