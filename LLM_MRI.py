import Treatment
import sys
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset, Features, Value, ClassLabel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import torch
class LLM_MRI:

    def __init__(self, model, device, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = torch.device(device)
        self.dataset = dataset
        self.base = Treatment(model, device)
        self.gridsize = 10 # Padrão
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
        
        # Adapting encodedDataset format
        self.dataset.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])

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
        
        # Selecting only lines containing the desired layer
        exclude = list(map(lambda x: True if x == category else False, datasetHiddenStates['label']))

        # Filtering the Dataset based on the desired label
        datasetByCategory = datasetHiddenStates.select(
            (
                i for i in range(len(datasetHiddenStates)) 
                if exclude[i] == True 
            )
        )

        # returning figure for the grid
        return self.base.plot_map(datasetByCategory, hidden_name, self.gridsize)



    def get_graph(self, category:int):
        '''
        Function that builds the pandas edgelist (graph representation) for the network region activations,
        for a given label (category) passed as a parameter.
        '''

        # Obtaining desired layer
        datasetHiddenStates = self.hidden_states_dataset
        
        # Selecting only lines containing the desired layer
        exclude = list(map(lambda x: True if x == category else False, datasetHiddenStates['label']))

        # Filtering the Dataset based on the desired label
        datasetByCategory = datasetHiddenStates.select(
            (
                i for i in range(len(datasetHiddenStates)) 
                if exclude[i] == True 
            )
        )

        hss = [x for x in datasetByCategory.column_names if x.startswith("hidden_state")]

        # crete an empty dataframe with columns up, down and corr
        df_graph = pd.DataFrame(columns=["cell_label_1", "cell_label_2", "weight", "level"])

        for hs in range(0, len(hss)-1):
            hs1 = hss[hs]
            hs2 = hss[hs+1]
            
            df_grid1 = self.base.get_grid(datasetByCategory, hs1, 10)
            df_grid2 = self.base.get_grid(datasetByCategory, hs2, 10)

            df_join = df_grid1[['cell_label']].join(df_grid2[['cell_label']], lsuffix='_1', rsuffix='_2')

            #group by and count the number of rows
            df_join_grouped = df_join.groupby(['cell_label_1', 'cell_label_2']).size().reset_index(name='weight')

            df_join_grouped['level'] = hs

            df_graph = pd.concat([df_graph, df_join_grouped])

            
        G = nx.from_pandas_edgelist(df_graph, 'cell_label_1', 'cell_label_2', ['weight'])

        return G


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

        nx.draw(g, pos, with_labels=True, node_size=2, node_color="skyblue", node_shape="o", alpha=0.9, linewidths=20)

        nx.draw_networkx_nodes(g,pos,
                            nodelist=nodelist,
                            node_size=150,
                            node_color='grey',
                            alpha=0.8)

        nx.draw_networkx_edges(g,pos,
                            edgelist = widths.keys(),
                            width=list(widths.values()),
                            edge_color='lightblue',
                            alpha=0.9)

        nx.draw_networkx_labels(g, pos=pos,
                                labels=dict(zip(nodelist,nodelist)),
                                font_color='black')

        return fig

sys.modules['LLM_MRI'] = LLM_MRI