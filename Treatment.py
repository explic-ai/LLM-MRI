import os #
import pandas as pd
import nltk #
from nltk.tokenize import word_tokenize #
import unicodedata #
from datasets import Dataset, Features, Value, ClassLabel
from transformers import AutoTokenizer, AutoModel
import torch #
from umap import UMAP #
from sklearn.preprocessing import MinMaxScaler #
import matplotlib.pyplot as plt
import numpy as np #
import seaborn as sns #
import sys

class Treatment:

    def __init__(self, model, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = device

    def setModel(self, model):
        '''
        Sets the model stored by the class.
        '''

        self.model = model

    def setDevice(self, device):
        '''
        Sets the device that will be used by the class.
        '''

        self.device = device
    
    def setTokenizer(self, tokenizer):
        '''
        Sets the tokenizer that will be used by the class.
        '''

        self.tokenizer = tokenizer

    def le_documentos(self, directory_path):
        '''
        Input: (string) Path to the directory where the corpus to be used is located.
        Output: (DataFrame) DataFrame corresponding to the corpus.

        This function iterates over the categories present in the document and stores them in a dataframe, separating the files present in each category.
        '''

        documentos = []

        for category in os.listdir(directory_path):
            for doc in os.listdir(f'{directory_path}/{category}'):
                name = doc.split('.')[0]
                path = f'{directory_path}/{category}/{doc}'
                content = open(path, "r", encoding="utf-8").read()
                documento = {"CATEGORY": category, "NAME": name, "PATH": path, "CONTENT": content}
                documentos.append(documento)

        df = pd.DataFrame(documentos)
        return df


    def remove_acentos(self, input_str):
        '''
        Input: (string) String representing the .txt file in the document.
        Output: (string) Input string with accents removed, according to Unicode.
        '''

        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


    def set_stopwords(self):
        '''
        Output: (list(string)) Stopwords, or words that limit the model's understanding,
        for the Portuguese language.
        '''

        stopwords = nltk.corpus.stopwords.words('portuguese')
        stopwords = [self.remove_acentos(palavra) for palavra in stopwords]
        return stopwords


    def normaliza_texto(self, txt, stopwords):
        '''
        Inputs: (string) String representing the .txt file to be normalized.
                (list(string)) Predefined stopwords in Portuguese.
        Output: (string) Normalized text corresponding to the file.
        '''

        return ' '.join([word for word in word_tokenize(str.lower(self.remove_acentos(txt))) if word not in stopwords and word.isalpha()])


    def set_normalizado(self, df, stopwords):
        '''
        Inputs: (DataFrame) DataFrame used to store the documents.
        Output: Adds a column with the normalized content to the DataFrame.
        '''

        df['CONTENT_NORMALIZADO'] = df.apply(lambda linha: self.normaliza_texto(str(linha['CONTENT']), stopwords), axis = 1)


    def DfToHuggingFacesDataset(self, df, class_names):
        '''
        Input: (DataFrame) DataFrame used to store the documents.
            (list(string)) List with the names of the classes.
        Output: (Dataset) DataFrame in the format of HuggingFace Dataset.
        '''

        # Pega os campos necessários
        df = df[['CONTENT_NORMALIZADO', 'CATEGORY']]
        df.columns = ['text', 'label']
        datasetFeatures = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
        dataset = Dataset.from_pandas(df, features = datasetFeatures)

        return dataset


    def tokenize(self, batch):
        '''
        Input: (Dataset) Dataset with text to be tokenized.
        Output: (Token) Tokenization of the Dataset, with padding enabled and a maximum length of 512.
        '''

        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


    def encodeDataset(self, dataset):
        '''
        Input: (Dataset) Dataset with text to be tokenized.
        Output: (Token) Tokenization of the Dataset, with padding enabled and a maximum length of 512.

        Maps over all items in the Dataset and performs tokenization.
        '''

        dataset_encoded = dataset.map(self.tokenize, batched=True, batch_size=None)
        return dataset_encoded


    def saveDataset(self, dataset):
        '''
        Input: (Dataset) Tokenized Dataset.

        Saves the Dataset to disk.
        '''

        dataset.save_to_disk("dataset_encoded.hf")


    def setEmbeddingsOnModel(self, model_ckpt):
        '''
        Input: (Model) Hugging Face model.
        Output: (Model) Model passed as a parameter.
        '''

        model = AutoModel.from_pretrained(model_ckpt).to(self.device)
        return model


    def getHiddenStatesOutputs(self, text, model):
        '''
        Inputs: (String) Text to be analyzed.
                (Model) Model used.
        Output: (Tensor) Network outputs for the input text.
        '''

        inputs = self.tokenizer(text, return_tensors = "pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states = True)
        return outputs


    def extract_hidden_states(self, batch):
        '''
        Input: (dict) Batch of data with model inputs.
        Output: (dict) Dictionary containing a tensor related to the extracted hidden layer weights and their respective labels.

        This function extracts only for the last layer of the model.
        '''


        inputs = {k:v.to(self.device) for k,v in batch.items() 
                if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


    def extract_all_hidden_states(self, batch):
        '''
        Input: (dict) Batch of data with model inputs.
        Output: (dict) Dictionary containing a tensor related to the extracted hidden layer weights and their respective labels.

        This function extracts for all hidden layers of the model.
        '''

        model = self.setEmbeddingsOnModel(model_ckpt=self.model)

        inputs = {k:v.to(self.device) for k,v in batch.items() 
                if k in self.tokenizer.model_input_names}
        
        with torch.no_grad():
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        all_hidden_states = {}
        
        for i, hs in enumerate(hidden_states):
            all_hidden_states[f"hidden_state_{i}"] = hs[:,0].cpu().numpy()
        
        return all_hidden_states


    def setDatasetToTorch(self, dataset_encoded):
        '''
        Input: (Dataset) Tokenized Dataset.
        Output: (Dataset) Dataset formatted for PyTorch.
        '''

        dataset_encoded.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])
        return dataset_encoded


    def get_embeddings(self, X, y):
        '''
        Input: (array) Matrix of embeddings for features X.
            (array) Embeddings vector for labels y.
        Output: (DataFrame) DataFrame with 2D embeddings.
        '''

        X_scaled = MinMaxScaler().fit_transform(X)
        mapper = UMAP(n_components=2, metric="cosine", random_state=42).fit(X_scaled)
        df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
        df_emb["label"] = y
        return df_emb


    def plot_map(self, dataset, hidden_state_label, map_dimension):
        '''
        Input: (Dataset) Dataset containing the hidden layers.
            (string) Hidden layer categories to be plotted.
        Output: Displays a 2D scatter plot of the embeddings.
        '''

        X = np.array(dataset[hidden_state_label])
        y = np.array(dataset["label"])
        df_emb = self.get_embeddings(X, y, map_dimension)
        fig, axes = plt.subplots(1, 2, figsize=(7,5))
        axes = axes.flatten()
        cmaps = ["Blues", "Reds"] # Depende do número de classes
        labels = dataset.features["label"].names

        for i, (label, cmap) in enumerate(zip(labels, cmaps)):
            df_emb_sub = df_emb.query(f"label == {i}")
            axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                        gridsize=10, linewidths=(0,)) # 
            axes[i].set_title(label)
            axes[i].set_xticks([]), axes[i].set_yticks([])

        fig.suptitle(hidden_state_label, fontsize=16)

        plt.tight_layout()
        plt.show()

   
    def plot_maps_all_hidden(self, dataset, dataset_all_hidden, map_dimension):
        '''
        Input: (Dataset) Dataset containing all hidden layers.
        Output: Plots the maps of all hidden layer embeddings.
        '''

        for hs in [x for x in dataset_all_hidden.column_names if x.startswith("hidden_state")]:
            print(hs)
            self.plot_map(dataset, hs, map_dimension)
    
   
    def hiddenStatesToNumpy(self, dataset_hidden):
        '''
        Input: (Dataset) Dataset containing all hidden layers.
        Output: (array, array) Numpy arrays with the hidden layer embeddings and labels.
        '''

        X = np.array(dataset_hidden["hidden_state"])
        y = np.array(dataset_hidden["label"])
        return X, y

    def get_grid(self, dataset, hidden_state_label, gridsize):
        '''
        Returns a dataframe containing the embeddings for each hidden layer of the network.
        '''

        hidden_state = hidden_state_label.split("_")[-1]
        X = np.array(dataset[hidden_state_label])
        y = np.array(dataset["label"])
        df_emb = self.get_embeddings(X, y)
        df_emb = df_emb.assign(
        X=pd.cut(df_emb.X, gridsize, labels=False),
        Y=pd.cut(df_emb.Y, gridsize, labels=False)
        )
        # create a new column with the concatenation of X and Y

        df_emb['cell_label'] = hidden_state + "_" + df_emb['X'].astype(str) + "_" + df_emb['Y'].astype(str)
        return df_emb
    
    def plot_all_grids(self, dataset, gridsize):
        '''
        Plots the grids for all embeddings of the hidden layers.
        '''

        for hs in [x for x in dataset.column_names if x.startswith("hidden_state")]:

            df_grid = self.get_grid(dataset, hs, gridsize)
            ct = pd.crosstab(df_grid.Y, df_grid.X, normalize=False)

            ct = ct.sort_index(ascending=False)
            
            sns.heatmap(ct, cmap="Blues", cbar=False, annot=True, fmt="d")
            #change figure title to hs
            plt.title(hs)
            plt.show()

sys.modules['Treatment'] = Treatment