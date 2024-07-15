import Treatment
import sys
from transformers import AutoTokenizer
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout
import torch

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

            colors = [list(mcolors.TABLEAU_COLORS.values())[i] for i in range(len(labels))]
            return colors
        else:
            return ['lightblue']


    def get_graph_image(self, G):
        """
        Displays the pandas edgelist (graph representation) for the network region activations,
        for a given graph passed as a parameter.

        Args:
            G (Graph): The networkx graph.

        Returns:
            fig (plt.figure): The matplotlib figure representation of the graph.
        """

        widths = nx.get_edge_attributes(G, 'weight')
        nodelist = G.nodes()

        pos = graphviz_layout(G, prog="dot")

        fig = plt.figure(figsize=(25,6))

        edge_colors = self.generate_graph_edge_colors(G)

        nx.draw(G, pos, with_labels=True, node_size=2, node_color="skyblue", node_shape="o", alpha=0.9, linewidths=20)

        nx.draw_networkx_nodes(G,pos,
                            nodelist=nodelist,
                            node_size=150,
                            node_color='grey',
                            alpha=0.8)

        nx.draw_networkx_edges(G,pos,
                            edgelist = widths.keys(),
                            width=list(widths.values()),
                            edge_color=edge_colors,
                            alpha=0.9)

        nx.draw_networkx_labels(G, pos=pos,
                                labels=dict(zip(nodelist,nodelist)),
                                font_color='black')

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

        g_composed = nx.compose(g1, g2)

        # Marking repeated edges
        duplicates = list(set(g1.edges) & set(g2.edges))
        for e in duplicates : g_composed.edges[e]['label'] = 2 
        
        return g_composed

sys.modules['LLM_MRI'] = LLM_MRI