import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch 
from umap import UMAP 
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from progress.bar import Bar
import networkx as nx

class Treatment:

    def __init__(self, model, device):
        """
        Initializes the Treatment class.

        Args:
            model (str): The model to be used.
            device (str): The device to be used (e.g., 'cpu' or 'cuda').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = device
        self.embeddings_dataset = []

    
    def get_embeddings_dataset(self):
        """
        Returns the embeddings dataset
        """
        return self.embeddings_dataset
    
    def tokenize(self, batch):
        """
        Tokenizes a batch of text.

        Args:
            batch (Dataset): Dataset with column "text" to be tokenized.

        Returns:
            Token: Tokenization of the Dataset, with padding enabled and a maximum length of 512.
        """
        if self.tokenizer.pad_token is None: # Adding eos as pad token for decoders
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer(batch["text"], padding = True, truncation=True, max_length=512)



    def encode_dataset(self, dataset):
        """
        Maps over all items in the Dataset and performs tokenization.

        Args:
            dataset (Dataset): Dataset with text to be tokenized.

        Returns:
            Token: Tokenization of the Dataset, with padding enabled and a maximum length of 512.
        """

        dataset_encoded = dataset.map(self.tokenize, batched=True, batch_size=None)
        return dataset_encoded


    def set_embeddings_on_model(self, model_ckpt):
        """
        Sets embeddings on the model.

        Args:
            model_ckpt (Model): Hugging Face model.

        Returns:
            Model: Model passed as a parameter.
        """

        model = AutoModel.from_pretrained(model_ckpt).to(self.device)

        return model


    def extract_all_hidden_states(self, batch):
        """
        Extracts all hidden states for a batch of data.

        Args:
            batch (dict): Batch of data with model inputs.

        Returns:
            dict: Dictionary containing a tensor related to the extracted hidden layer weights and their respective labels.
        """
        
        model = self.set_embeddings_on_model(model_ckpt=self.model)

        inputs = {k:v.to(self.device) for k,v in batch.items() 
                if k in self.tokenizer.model_input_names}
        
        with torch.no_grad():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        all_hidden_states = {}
        
        for i, hs in enumerate(hidden_states):
            all_hidden_states[f"hidden_state_{i}"] = hs[:,0].cpu().numpy()
        
        return all_hidden_states


    def set_dataset_to_torch(self, dataset_encoded):
        """
        Sets the dataset format to PyTorch.

        Args:
            dataset_encoded (Dataset): Tokenized Dataset.

        Returns:
            Dataset: Dataset formatted for PyTorch.
        """

        dataset_encoded.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])
        return dataset_encoded

    def svd_graph(self, dataset_hidden_states, dim=40):

        #1) Para cada hidden state: (dicionário de tensores)
        #   Reduzir a dimensionalidade para 100 dimensões

        # Criar um grafo do networkx

        #2) Para cada conexão de hidden state:
        #   Calcular a matriz de correlação, e salvar como df em outra variável da classe
        #   Salvar cada coluna / linha com um nome específico identificável
        #   (corr_matrix = pd.DataFrame(data, columns=column_names, index=row_names))
        #   Adicionar cada linha e coluna como nó no grafo

        #3) Para cada matriz de correlação: 
        #   Para cada par i, j em linha, coluna:
        #       Criar uma aresta entre o primeiro e segundo nó, com peso = correlação
        #       Colocar 'label' = 0 como atributo da aresta, por enquanto

        # Retornar o grafo

        reduced_hs_list = []
        
        # 1) Reducing dimensionality through SVD
        for hs_name in [x for x in dataset_hidden_states.column_names if x.startswith("hidden_state")]:

            # dataset_hidden_states[hs_name] = dataset_hidden_states[hs_name].to(self.device)
            U, s, Vt = torch.linalg.svd(dataset_hidden_states[hs_name], full_matrices = False)
            print("Dataset shape: ", dataset_hidden_states[hs_name].shape)
            print("U shape: ", U.shape)
            print("s shape: ", s.shape)
            print("Vt shape: ", Vt.shape)
            print('---------------------------------------------------------')

            # Choosing the "dim" main components
            U_k = U[:, :dim]  # Keep first k columns of U (40 x 100)
            s_k = s[:dim]

            print("s_k shape: ", s_k.shape)
            print((s_k))
            # Multiplying to obtain the reduced dataset
            reduced_hs = U_k @ torch.diag(s_k)
            print(type(reduced_hs))
            reduced_hs_list.append(reduced_hs)
            print("reduced_shape: ", reduced_hs.shape)
        
        # Creating the graph
        G = nx.Graph()
        print("reduced list: ", len(reduced_hs_list))

        # Variable to store the correlation matrices
        correlation_reduced_hs = []

        # 2) Calculating correlation for every hidden state intersection
        for index in range(len(reduced_hs_list) - 1):
            first_layer = reduced_hs_list[index]
            second_layer = reduced_hs_list[index+1]

            # Compute the dot product between X and Y
            dot_product = first_layer.T @ second_layer

            # Compute the norms of X and Y
            norm_X = torch.norm(first_layer, p=2, dim=0, keepdim=True)
            norm_Y = torch.norm(second_layer, p=2, dim=0, keepdim=True)

            # Compute the cosine similarity matrix
            cosine_similarity_matrix = dot_product / (norm_X.T @ norm_Y)

            # Generating names for columns and rows (hs{x}_{index})
            column_names = [f'{index}_{x}' for x in range(dim)]
            row_names = [f'{index+1}_{x}' for x in range(dim)]

            # Adding all different nodes to the graph
            G.add_nodes_from(column_names)
            G.add_nodes_from(row_names)

            # Turning matrix into DataFrame, so that components can be named
            cosine_matrix_df = pd.DataFrame(cosine_similarity_matrix.detach().numpy(), columns=column_names, index=row_names)

            # Storing matrix
            correlation_reduced_hs.append(cosine_matrix_df)
        
        # 3) Adding edges to the graph
        for corr_matrix in correlation_reduced_hs:
            for row_name, row_data in corr_matrix.iterrows(): # Iterating though rows
                for col_name, weight in row_data.items(): # Iterating through columns
                    if weight > 0.3:
                        G.add_edge(col_name, row_name, weight=weight * 3, label=0) # Adding edges
        
        # Returning the full graph developed
        return G

        

    def get_embeddings(self, X, y):
        """
        Gets 2D embeddings by reducing the datasets dimensionalty for features and labels.

        Args:
            X (array): Matrix of embeddings for features.
            y (array): Embeddings vector for labels.

        Returns:
            DataFrame: DataFrame with 2D embeddings.
        """
        X_scaled = MinMaxScaler().fit_transform(X)

        mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled) #, random_state=42
        df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
        df_emb["label"] = y
        return df_emb


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
        df_emb = self.get_embeddings(X, y)

        # Saving the embeddings of those documents
        self.embeddings_dataset.append(df_emb)

        # Turning into gridsize x gridsize
        df_emb = df_emb.assign(
        X=pd.cut(df_emb.X, gridsize, labels=False),
        Y=pd.cut(df_emb.Y, gridsize, labels=False)
        )

        # Adjusting Labels
        df_emb['cell_label'] = hidden_state + "_" + df_emb['X'].astype(str) + "_" + df_emb['Y'].astype(str)
        return df_emb


    def get_activations_grid(self, hidden_layer_name, label, label_name, df_grid):
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

        df_grid = df_grid.loc[df_grid['label'] == label]

        ct = pd.crosstab(df_grid.Y, df_grid.X, normalize=False)

        ct = ct.sort_index(ascending=False)
        
        fig = sns.heatmap(ct, cmap="Blues", cbar=False, annot=True, fmt="d")
        
        #change figure title to hs
        full_name = f"{hidden_layer_name} : {label_name}"
        plt.title(full_name)
        
        return fig

    def get_all_grids(self, dataset, gridsize, buffer):
        """
        Gets and stores all dimension-reduced grids on buffer 

        Args:
            dataset (Dataset): The dataset to be used.
            gridsize (int): The grid size.
            buffer (list): Buffer to store grids

        Returns:
            buffer
        """
        with Bar('Processing layers...', max=len([x for x in dataset.column_names if x.startswith("hidden_state")])) as bar:
            for hs in [x for x in dataset.column_names if x.startswith("hidden_state")]:
                df_grid = self.get_grid(dataset, hs, gridsize)
                bar.next()
                buffer.append(df_grid) # ith grid
        
        return buffer
