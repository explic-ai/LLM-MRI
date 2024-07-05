import Treatment
import sys
from transformers import AutoTokenizer
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout
import torch
from datasets import features, ClassLabel
class LLM_MRI:

    def __init__(self, model, device, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = torch.device(device)
        self.dataset = dataset
        self.base = Treatment(model, device)
        self.gridsize = 10 # Padrão
        self.class_names = self.dataset.features['label'].names
        self.dataset = self.initialize_dataset() 
        self.hidden_states_dataset = ""
        
    def setDevice(self, device):
        '''
        Sets the device that will be used by the class.
        '''

        self.device = torch.device(device)

    def setDataset(self, dataset):
        '''
        Sets the dataset that will be used by the class.
        '''

        self.dataset = dataset

    def initialize_dataset(self):
        '''
        Initializes the encoded dataset from the model and transforms it into the torch type.
        '''
        encodedDataset = self.base.encodeDataset(self.dataset)

        # Transformando o dataset para o formato Torch
        encodedDataset = self.base.setDatasetToTorch(encodedDataset)
        return encodedDataset

    def process_activation_areas(self, map_dimension:int):
        '''
        Input: (int) Size of the side of the square that will show the visualization.

        Output: None, sets the self.hidden_states_dataset as the datasetHiddenStates
        '''

        # Getting self.dataset Hidden Layers
    
        datasetHiddenStates = self.dataset.map(self.base.extract_all_hidden_states, batched=True)

        self.gridsize = map_dimension
        self.hidden_states_dataset = datasetHiddenStates


    def get_layer_image(self, layer:int, category:int):
        '''
        Input: (int) Layer to be visualized
            (int) Category whose activations will be visualized

        Output: Plot of the activation grid for the desired layer and category.
        '''

        # Obtaining layer passed by parameter
        datasetHiddenStates = self.hidden_states_dataset

        # Obtaining layer name string 
        hidden_name = f"hidden_state_{layer}"

        # Selecting layer passed as parameter
        datasetHiddenStates = datasetHiddenStates.remove_columns(
            [col for col in datasetHiddenStates.column_names 
            if col not in [hidden_name, 'text', 'label', 'input_ids', 'attention_mask']]
        )

        category_to_int = self.class_names.index(category)
       
        return self.base.get_activations_grid(datasetHiddenStates, self.gridsize, hidden_name, category_to_int, category)
       


    def get_graph(self, category:bool = false):
        '''
        Function that builds the pandas edgelist (graph representation) for the network region activations,
        for a given label (category) passed as a parameter.
        '''

        # Obtaining desired layer
        datasetHiddenStates = self.hidden_states_dataset
        
        hss = [x for x in datasetHiddenStates.column_names if x.startswith("hidden_state")]

        # crete an empty dataframe with columns up, down and corr
        df_graph = pd.DataFrame(columns=["cell_label_1", "cell_label_2", "weight", "level"])

        for hs in range(0, len(hss)-1):
            hs1 = hss[hs]
            hs2 = hss[hs+1]
            
            df_grid1 = self.base.get_grid(datasetHiddenStates, hs1, self.gridsize)
            df_grid2 = self.base.get_grid(datasetHiddenStates, hs2, self.gridsize)

            # when no category is passed gets all values
            if category == 2:
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
        if category != 2:
            nx.set_edge_attributes(G, category, "label")

        return G

    def generate_graph_edge_colors(G):
        '''
        Function that generates a list of colors based on the amount of labels in the graphs edges.
        If no label is present, it returns a default value.
        '''
        edge_attributes = list(list(G.edges(data=True))[0][-1].keys())
        
        if 'label' in edge_attributes:
            labels = list(set(nx.to_pandas_edgelist(g_composed)['label']))

            colors = [list(mcolors.TABLEAU_COLORS.values())[i] for i in range(len(labels))]
            return colors
        else:
            return ['lightblue']

    def get_graph_image(self, category:int):
        '''
        Function that displays the pandas edgelist (graph representation) for the network region activations,
         for a given label (category) passed as a parameter.
        '''

        g = self.get_graph(category)

        widths = nx.get_edge_attributes(g, 'weight')
        nodelist = g.nodes()

        pos = graphviz_layout(g, prog="dot")

        fig = plt.figure(figsize=(25,6))

        edge_colors = self.generate_graph_edge_colors(g)

        nx.draw(g, pos, with_labels=True, node_size=2, node_color="skyblue", node_shape="o", alpha=0.9, linewidths=20)

        nx.draw_networkx_nodes(g,pos,
                            nodelist=nodelist,
                            node_size=150,
                            node_color='grey',
                            alpha=0.8)

        nx.draw_networkx_edges(g,pos,
                            edgelist = widths.keys(),
                            width=list(widths.values()),
                            edge_color=edge_colors,
                            alpha=0.9)

        nx.draw_networkx_labels(g, pos=pos,
                                labels=dict(zip(nodelist,nodelist)),
                                font_color='black')

        return fig

sys.modules['LLM_MRI'] = LLM_MRI