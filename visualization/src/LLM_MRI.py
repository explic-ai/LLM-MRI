from Treatment import Treatment
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset, Features, Value, ClassLabel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import torch

class LLM_MRI:

    def __init__(self, model, device, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = torch.device(device)
        self.dataset = dataset
        self.base = Treatment(model, device)
        self.gridzise = 10 # Padrão

    def setDevice(self, device):
        '''
        Define o dispositivo (device) que será utilizado pela classe
        '''
        self.device = torch.device(device)

    def setDataset(self, dataset):
        '''
        Define o dataset que será utilizado pela classe
        '''
        self.dataset = dataset

    def initialize_dataset(self):
        '''
        Inicializando o encoded Dataset a partir do modelo, 
        e o transformando no tipo torch
        '''
        encodedDataset = self.base.encodeDataset(self.dataset)

        # Transformando o dataset para o formato Torch
        encodedDataset = self.base.setDatasetToTorch(encodedDataset)
        return encodedDataset

    def process_activation_areas(self, map_dimension):
        '''
        Entrada: (int) Tamanho do lado do quadrado que será mostrada
        a visualização

        Saída: Plot do grid de ativações para cada uma das camadas 
        do modelo
        '''

        # Obtendo o encoded Dataset
        encodedDataset = self.initialize_dataset()
        print(type(encodedDataset), " TIPO DO ENCODED")

        # Adapting encodedDataset format
        encodedDataset.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])

        # Obtendo as camadas ocultas do Dataset
        datasetHiddenStates = encodedDataset.map(self.base.extract_all_hidden_states, batched=True)

        # Plotando as visualizações
        self.base.plot_all_grids(datasetHiddenStates, map_dimension)


    def get_layer_image(self, layer, category):
        '''
        Entrada: (int) Camada que será visualizada
                 (int) Categoria cujas ativações serão visualizada
        
        Saída: Plot do grid de ativações para a camada e categoria desejada
        '''
        
        # Obtendo o encodede Dataset
        encodedDataset = self.initialize_dataset()

        # Adapting encodedDataset format
        encodedDataset.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])

        # Obtendo o layer desejado
        datasetHiddenStates = encodedDataset.map(self.base.extract_all_hidden_states, batched=True)

        # Obtendo a string referente ao nome da camada
        hidden_name = f"hidden_state_{layer}"

        # Selecionando apenas a camada passada por parâmetro
        datasetHiddenStates = datasetHiddenStates.remove_columns(
            [col for col in datasetHiddenStates.column_names 
            if col not in [hidden_name, 'text', 'label', 'input_ids', 'attention_mask']]
        )
        
        
        # Selecionando as linhas que contém a label da categoria desejada
        exclude = list(map(lambda x: True if x == category else False, datasetHiddenStates['label']))

        # Filtrando o Dataset com base nas labels obtidas
        datasetByCategory = datasetHiddenStates.select(
            (
                i for i in range(len(datasetHiddenStates)) 
                if exclude[i] == True 
            )
        )

        # Plotando a visualização
        self.base.plot_all_grids(datasetByCategory, 10)


    def get_all_graph(self):
        '''
        Função que constrói o pandas edgelist (representação em grafos) para as ativações das regiões da rede. 
        '''

        # def make_graph(dataset, gridsize, min_corr=0.5):
        hss = [x for x in self.dataset.column_names if x.startswith("hidden_state")]

        # crete an empty dataframe with columns up, down and corr
        df_graph = pd.DataFrame(columns=["cell_label_1", "cell_label_2", "weight", "level"])

        for hs in range(0, len(hss)-1):
            hs1 = hss[hs]
            hs2 = hss[hs+1]

            print(hs1, hs2)
            
            df_grid1 = self.base.get_grid(self.dataset, hs1, self.gridsize)
            df_grid2 = self.base.get_grid(self.dataset, hs2, self.gridsize)

            df_join = df_grid1[['cell_label']].join(df_grid2[['cell_label']], lsuffix='_1', rsuffix='_2')

            #group by and count the number of rows
            df_join_grouped = df_join.groupby(['cell_label_1', 'cell_label_2']).size().reset_index(name='weight')

            df_join_grouped['level'] = hs

            df_graph = pd.concat([df_graph, df_join_grouped])

        # make a Networkx graph from df_graph
            
        G = nx.from_pandas_edgelist(df_graph, 'cell_label_1', 'cell_label_2', ['weight'])

        return G
            

    def get_graph(self, category):
        '''
        Função que constrói o pandas edgelist (representação em grafos) para as ativações das regiões da 
        rede, para uma determinada label (categoria) passada como parâmetro. 
        '''

        # Selecionando as linhas que contém a label da categoria desejada
        exclude = list(map(lambda x: True if x == category else False, self.dataset['label']))

        # Filtrando o Dataset com base nas labels obtidas
        datasetByCategory = self.dataset.select(
            (
                i for i in range(len(self.dataset)) 
                if exclude[i] == True 
            )
        )

        hss = [x for x in datasetByCategory.column_names if x.startswith("hidden_state")]

        # crete an empty dataframe with columns up, down and corr
        df_graph = pd.DataFrame(columns=["cell_label_1", "cell_label_2", "weight", "level"])

        for hs in range(0, len(hss)-1):
            hs1 = hss[hs]
            hs2 = hss[hs+1]

            print(hs1, hs2)
            
            df_grid1 = self.base.get_grid(datasetByCategory, hs1, self.gridsize)
            df_grid2 = self.base.get_grid(datasetByCategory, hs2, self.gridsize)

            df_join = df_grid1[['cell_label']].join(df_grid2[['cell_label']], lsuffix='_1', rsuffix='_2')

            #group by and count the number of rows
            df_join_grouped = df_join.groupby(['cell_label_1', 'cell_label_2']).size().reset_index(name='weight')

            df_join_grouped['level'] = hs

            df_graph = pd.concat([df_graph, df_join_grouped])

        # make a Networkx graph from df_graph
            
        G = nx.from_pandas_edgelist(df_graph, 'cell_label_1', 'cell_label_2', ['weight'])

        return G


    def get_graph_image(self, category):
        '''
        Função que mostra na tela o pandas edgelist (representação em grafos) para as ativações das regiões da 
        rede, para uma determinada label (categoria) passada como parâmetro. 
        '''

        g = self.get_graph(category)
        widths = nx.get_edge_attributes(g, 'weight')
        nodelist = g.nodes()

        pos = graphviz_layout(g, prog="dot")
        #make larger figure
        plt.figure(figsize=(25,6))


        # nx.draw(g, pos, with_labels=True, node_size=2, node_color="skyblue", node_shape="o", alpha=0.9, linewidths=20)

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

        plt.box(False)
        plt.show()
