from ._base import Graph
import pandas as pd
from typing import Union
import networkx as nx
from progress.bar import Bar
import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

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
        self.grids = self.get_all_grids(self.hidden_states, gridsize=10)
        self.gridsize = self.gridsize # Aqui, tem o problema dos n² nrags
        
    def get_grid(self, dataset, hidden_state_label, gridsize):
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
                df_grid = self.get_grid(dataset, hs, gridsize)
                bar.next()
                buffer.append(df_grid)  # ith grid

        return buffer
    
    def build_graph(self):
        """
        Builds the pandas edgelist (graph representation) for the network region activations,
        for a given label (category) passed as a parameter.

        Args:
            category_name (str): The name of the category. Default is an empty string.

        Returns:
            Graph: The networkx graph representing the activations.
        """
        graphs_list = []

        for category_name in self.categories.keys():

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

        return g_composed
    