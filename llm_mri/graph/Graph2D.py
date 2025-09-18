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

    def __init__(self, n_components:int, hidden_states: dict, num_layers: int, gridsize: int = 10, class_names: list = None, reduction_method: object = None):
        """
        Initialize the GraphND object.

        :param n_components: The number of components that the graph was reduced based on.
        """
        super().__init__(n_components, hidden_states, reduction_method, class_names, num_layers)
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

        label = self.class_names.index(category_name)

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
        Builds the networkx graph for the network region activations,
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
                category_idx = self.class_names.index(category_name)

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
                nx.set_edge_attributes(G, values=category_idx, name="label")

            graphs_list.append(G)

        # compose without relabeling to 0/1/2
        if len(graphs_list) > 1:
            g1, g2 = graphs_list[0], graphs_list[1]
            g_composed = nx.compose(g1, g2)

            # mark overlaps; keep existing label (class index)
            duplicates = set(g1.edges()) & set(g2.edges())
            for e in duplicates:
                g_composed.edges[e]['overlap'] = True
        else:
            g_composed = graphs_list[0]

        g_composed.graph['label_names'] = category_list
        g_composed.graph['num_layers'] = self.num_layers

        return g_composed

    # Instanciar esse método na classe grafo
    def _get_node_positions(self, G, fix_node_positions):
        
        # Verifying if graph passed is a networkx graph
        if not isinstance(G, nx.Graph):
            raise TypeError("The graph must be a networkx Graph object.")
        
        # Verifying if the graph has nodes
        if G.number_of_nodes() == 0:
            raise ValueError("The graph has no nodes to display.")
        
        # Verifying if the graph has edges
        if G.number_of_edges() == 0:
            raise ValueError("The graph has no edges to display.")

        # By default, full_graph is the current graph
        full_graph = G

        if fix_node_positions: # If asked to fix, the graph of all categories will be considered
            
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
            height_index = self.num_layers - int(node.split('_')[0]) 
            new_pos[node] = (pos[node][0], height_index)


        return new_pos