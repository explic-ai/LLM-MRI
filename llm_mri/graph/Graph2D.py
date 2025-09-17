from ._base import Graph
import pandas as pd
from typing import Union, List
import networkx as nx
from progress.bar import Bar
import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

class Graph2D(Graph):

    def __init__(self, n_components:int, categories:dict, hidden_states: dict, num_layers: int, gridsize: int = 10):
        """
        Initialize the GraphND object.

        :param n_components: The number of components that the graph was reduced based on.
        """
        super().__init__(n_components, categories)
        self.n_components = n_components
        self.categories = [categories] if isinstance(categories, str) else categories
        self.hidden_states = hidden_states
        self.num_layers = num_layers
        self.gridsize = gridsize # Aqui, tem o problema dos n² nrags
        self.reduced_dataset = self.get_all_grids(self.hidden_states)
        self.full_graph = self.build_graph() # Grafo com todas as categorias
        

    def get_grid_image(self, layer, category_name):
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

        # Chooses the specific layer
        df_grid = self.reduced_dataset[layer]

        label = self.categories[category_name]

        df_grid = df_grid.loc[df_grid['label'] == label]

        ct = pd.crosstab(df_grid.Y, df_grid.X, normalize=False)

        ct = ct.sort_index(ascending=False)

        fig = sns.heatmap(ct, cmap="Blues", cbar=False, annot=True, fmt="d")

        # change figure title to hs
        full_name = f"Hidden State {layer} : {category_name}"
        plt.title(full_name)

        return fig


    def build_grid(self, dataset, hidden_state_label, gridsize):
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

        # --- inlined get_embeddings (unchanged logic) ---
        X_scaled = MinMaxScaler().fit_transform(X)
        mapper = UMAP(n_components=2, metric="cosine").fit(
            X_scaled)  # , random_state=42
        df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
        df_emb["label"] = y
        # ------------------------------------------------

        # Turning into gridsize x gridsize
        df_emb = df_emb.assign(
            X=pd.cut(df_emb.X, gridsize, labels=False),
            Y=pd.cut(df_emb.Y, gridsize, labels=False)
        )

        # Adjusting Labels
        df_emb['cell_label'] = hidden_state + "_" + \
            df_emb['X'].astype(str) + "_" + df_emb['Y'].astype(str)
        return df_emb
    

    def get_all_grids(self, dataset): 
        """
        Gets and stores all dimension-reduced grids on buffer 

        Args:
            dataset (Dataset): The dataset to be used.
            gridsize (int): The grid size.
            buffer (list): Buffer to store grids

        Returns:
            buffer
        """

        buffer = []
        gridsize = self.gridsize

        # aqui, usa o self.hidden_states_dataset
        with Bar('Processing layers...', max=len([x for x in dataset.column_names if x.startswith("hidden_state")])) as bar:
            for hs in [x for x in dataset.column_names if x.startswith("hidden_state")]:
                df_grid = self.build_grid(dataset, hs, gridsize)
                bar.next()
                buffer.append(df_grid)  # ith grid

        return buffer

    
    def build_graph(self, category_list: Union[str, List[str]]=None):
        """
        Builds the pandas edgelist (graph representation) for the network region activations,
        for a given label (category) passed as a parameter.

        Args:
            category_name (str): The name of the category. Default is an empty string.

        Returns:
            Graph: The networkx graph representing the activations.
        """
        graphs_list = []

        if category_list is None:
            category_list = [""]

        if isinstance(category_list, str):
            category_list = [category_list]

        for category_name in category_list:

            if(category_name != ""):
                category_idx = self.categories[category_name]

            else:
                category_idx = -1
            
            # Obtaining desired layer
            datasetHiddenStates = self.hidden_states # aqui, usa o self.hidden_states_dataset
            
            hss = [x for x in datasetHiddenStates.column_names if x.startswith("hidden_state")]

            # crete an empty dataframe with columns up, down and corr
            df_graph = pd.DataFrame(columns=["cell_label_1", "cell_label_2", "weight", "level"])

            for hs in range(0, len(hss)-1):
                
                # Aqui, já tinha sido reduzido por UMAP (chamava o get_all_grids)
                df_grid1 = self.reduced_dataset[hs]
                df_grid2 = self.reduced_dataset[hs+1]

                # when no category_idx is passed gets all values
                if category_idx == -1:
                    df_join = df_grid1[['cell_label']].join(df_grid2[['cell_label']], lsuffix='_1', rsuffix='_2')
                
                # when category_idx is passed filters values by category_idx
                else:
                    df_join = df_grid1.loc[df_grid1['label'] == category_idx][['cell_label']].join(df_grid2.loc[df_grid2['label'] == category_idx][['cell_label']], lsuffix='_1', rsuffix='_2')

                # group by and count the number of rows
                df_join_grouped = df_join.groupby(['cell_label_1', 'cell_label_2']).size().reset_index(name='weight')

                df_join_grouped['level'] = hs

                df_graph = pd.concat([df_graph, df_join_grouped])

                
            G = nx.from_pandas_edgelist(df_graph, 'cell_label_1', 'cell_label_2', ['weight'])

            # when generating category_idx graph, assigns category_idx label to edges
            if category_idx != -1:
                nx.set_edge_attributes(G, category_idx, "label")

            graphs_list.append(G)

        # Adjusting formatting in case it has more than one category
        if (len(graphs_list) > 1):
            g1 = graphs_list[0]
            g2 = graphs_list[1]

            for u, v, data in g1.edges(data=True):
                data['label'] = 0
            
            for u, v, data in g2.edges(data=True):
                data['label'] = 1

            g_composed = nx.compose(g1, g2)

            # Marking repeated edges
            duplicates = list(set(g1.edges) & set(g2.edges))
            for e in duplicates : g_composed.edges[e]['label'] = 2 
        
        else:
            g_composed = graphs_list[0]
        
        g_composed.graph['label_names'] = category_list
        g_composed.graph['num_layers'] = self.num_layers

        return g_composed

    # Need to specify the generate node colors and generate edge colors methods
    def get_graph_image(self, G, colormap = 'coolwarm', fix_node_positions:bool = True):
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

            # Getting the graph with all possible activations
                full_graph = self.full_graph


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