from llm_mri.Treatment import Treatment
from transformers import AutoTokenizer
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize

class LLM_MRI:

    def __init__(self, model, device, dataset):
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

    def set_device(self, device):
        """
        Sets the device that will be used by the class.

        Args:
            device (str): The device to be used (e.g., 'cpu' or 'cuda').
        """
        self.device = torch.device(device)

    def set_dataset(self, dataset):
        """
        Sets the dataset that will be used by the class.

        Args:
            dataset (Dataset): The dataset to be used.
        """

        self.dataset = dataset

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
        self.reduced_dataset = self.base.get_all_grids(datasetHiddenStates, map_dimension, self.reduced_dataset)


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

            #group by and count the number of rows
            df_join_grouped = df_join.groupby(['cell_label_1', 'cell_label_2']).size().reset_index(name='weight')

            df_join_grouped['level'] = hs

            df_graph = pd.concat([df_graph, df_join_grouped])

            
        G = nx.from_pandas_edgelist(df_graph, 'cell_label_1', 'cell_label_2', ['weight'])

        # when generating category graph, assigns category label to edges
        if category != -1:
            nx.set_edge_attributes(G, category, "label")

        return G


    def generate_graph_edge_colors(self, G):
        """
        Generates a list of colors based on the amount of labels in the graph's edges.

        Args:
            G (Graph): The networkx graph.

        Returns:
            list: A list of colors for the graph's edges.
        """
        edge_attributes = list(list(G.edges(data=True))[0][-1].keys())
        
        if 'label' in edge_attributes:
            
            labels = list(set(nx.to_pandas_edgelist(G)['label']))

            colors = [list(mcolors.TABLEAU_COLORS.values())[i] for i in range(min(2, len(labels)))]

            if(len(colors) > 1):
                rgb1 = mcolors.to_rgb(colors[0])
                rgb2 = mcolors.to_rgb(colors[1])
                
                # Calculate blended color
                colors.append(mcolors.to_hex(tuple(0.5 * c1 + (1 - 0.5) * c2 for c1, c2 in zip(rgb1, rgb2))))

            return colors

        else:
            return ['lightblue']


    def get_graph_image(self, G):
        """
        Generates a matplotlib figure of the graph with nodes as pizza graphics.
        Args:
        G (networkx.Graph): The NetworkX graph.
        Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure representing the graph.
        """


        # Get all nodes
        nodelist = list(G.nodes())
        
        # If your edges have a 'weight' attribute, otherwise this will be empty
        widths = nx.get_edge_attributes(G, 'weight')
        
        # Use graphviz_layout for positioning
        pos = graphviz_layout(G, prog="dot")
        
        # Adjust the y-coordinates based on node identifiers (assuming they start with a digit)
        heights = sorted(list(set([x[1] for x in pos.values()])), reverse=True)
        
        new_pos = {}
        for node in pos:

            # Extract the first character to determine height index
            height_index = int(node.split('_')[0])  # Adjust based on your node naming convention
            new_pos[node] = (pos[node][0], heights[height_index])
        pos = new_pos
        
        # Create the matplotlib figure
        fig, ax = plt.subplots(figsize=(25, 6))
        
        # Generate edge colors
        edge_colors = self.generate_graph_edge_colors(G)

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
                for u, v in widths.keys()]
            
        # Add legend labels
        self.label_names.append("both")
        
        # Create legend handles based on edge colors
        legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in (edge_colors)]
        plt.legend(legend_handles, self.label_names, loc='upper right')
        
        # Compute the degree of each node
        degrees = dict(G.degree())
        
        # Scale node sizes
        max_degree = max(degrees.values())
        node_sizes = [100 + (degrees[node] / max_degree) * 1400 for node in nodelist]

        # Coloring Nodes
        node_colors = self.generate_node_colors(G, edge_colors)

        # Draw edges with specified widths and colors
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=widths.keys(),
            width=[widths[edge] for edge in widths],
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
            edgecolors='black'  # Optional: Adds a border to nodes
        )
        
        # Draw labels for nodes
        nx.draw_networkx_labels(
            G,
            pos,
            labels={node: node for node in nodelist},
            font_color='black',
            font_size=10,
            ax=ax
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

    
    def generate_node_colors(self, G, edge_colors):

        """
        Generates a list of colors based on the amount of nodes in the graph's edges, being
        each node color's proportional to the amount of times the node was activated by a label.

        Args:
            G (Graph): The networkx graph.
            edge_colors (list): List containing the edge colors

        Returns:
            list: A list of colors for the graph's nodes.
        """
        cont_2 = 0
        
        if len(self.label_names) > 2:

            label_0_counts = {node: 0 for node in G.nodes()}
            label_1_counts = {node: 0 for node in G.nodes()}

            for u, v, data in G.edges(data=True):

                label = data.get('label')
                weight = data.get('weight', 1)

                if label == 0:
                    label_0_counts[u] += weight
                    label_0_counts[v] += weight

                elif label == 1:
                    label_1_counts[u] += weight
                    label_1_counts[v] += weight

                elif label == 2:
                    cont_2 += 1
                    label_0_counts[u] += weight
                    label_1_counts[u] += weight
                    label_0_counts[v] += weight
                    label_1_counts[v] += weight 
            
    

            cmap = LinearSegmentedColormap.from_list('custom_gradient', [edge_colors[0], edge_colors[1]])

            # Normalize the ratios between 0 and 1
            norm = Normalize(vmin=0, vmax=1)

            # Calculate the proportion of label_1 edges for each node
            proportions = []
            for node in G.nodes():
                label0 = label_0_counts.get(node, 0)
                label1 = label_1_counts.get(node, 0)
                total = label0 + label1
                ratio = (label1 / total) if total > 0 else 0.5  # Neutral ratio if no edges
                proportions.append(ratio)

            # Map proportions to colors using the colormap
            node_colors = [cmap(norm(ratio)) for ratio in proportions]
        
        else:
            node_colors = 'gray'

        return node_colors

