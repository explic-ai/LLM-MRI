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

class Treatment:

    def __init__(self, model, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.device = device

    def setModel(self, model):
        '''
        Define o modelo guardado pela classe
        '''
        self.model = model

    def setDevice(self, device):
        '''
        Define o dispositivo (device) que será utilizado pela classe
        '''
        self.device = device
    
    def setTokenizer(self, tokenizer):
        '''
        Define o tokenizer que será utilizado pela classe
        '''
        self.tokenizer = tokenizer

    def le_documentos(self, directory_path):
        '''
        Entrada: (string) Path para o diretório onde está o corpus a ser utilizado
        Saída: (DataFrame) DataFrame correspondente ao corpus

        Essa função itera pelas categorias presentes no documento,
        e as guarda num dataframe, de modo a separar os arquivos presentes
        em cada categoria
        
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
        Entrada: (string) String referente ao arquivo .txt no documento
        Saída: (string) String da entrada, removendo os acentos, conforme o unicoe
        '''
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


    def set_stopwords(self):
        '''
        Saída: (list(string)) Stopwords, ou palavras que limitam a compreenão
        do modelo, para a língua portuguesa
        '''
        stopwords = nltk.corpus.stopwords.words('portuguese')
        stopwords = [self.remove_acentos(palavra) for palavra in stopwords]
        return stopwords


    def normaliza_texto(self, txt, stopwords):
        '''
        Entradas: (string) String referente ao arquivo .txt a ser normalizado
                  (list(string)) Stopwords em português previamente definidas
        Saída: (string) Texto normalizado referente ao arquivo 
        '''
        return ' '.join([word for word in word_tokenize(str.lower(self.remove_acentos(txt))) if word not in stopwords and word.isalpha()])


    def set_normalizado(self, df, stopwords):
        '''
        Entradas: (DataFrame) DataFrame utilizado para guardar os documentos
        Saída: adiciona uma coluna com o conteúdo normalizado ao DataFrame
        '''
        df['CONTENT_NORMALIZADO'] = df.apply(lambda linha: self.normaliza_texto(str(linha['CONTENT']), stopwords), axis = 1)
        # print(df['CONTENT_NORMALIZADO'].head())


    def DfToHuggingFacesDataset(self, df, class_names):
        '''
        Entrada: (DataFrame) DataFrame utilizado para guardar os documentos
                 (list(string)) Lista com o nome das classes
        Saída: (Dataset) DataFrame com formato de HuggingFace Dataset
        '''
        # Pega os campos necessários
        df = df[['CONTENT_NORMALIZADO', 'CATEGORY']]
        df.columns = ['text', 'label']
        datasetFeatures = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
        dataset = Dataset.from_pandas(df, features = datasetFeatures)

        return dataset


    def tokenize(self, batch):
        '''
        Entrada: (Dataset) Dataset com texto para ser tokenizado
        Saída: (Token) Tokenização do Dataset, com o padding ativado e tamanho máximo de 512
        '''
        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


    def encodeDataset(self, dataset):
        '''
        Entrada: (Dataset) Dataset com texto para ser tokenizado
        Saída: (Token) Tokenização do Dataset, com o padding ativado e tamanho máximo de 512

        Mapeia por todos os itens do Dataset, e realiza a tokenização
        '''
        dataset_encoded = dataset.map(self.tokenize, batched=True, batch_size=None)
        return dataset_encoded


    def saveDataset(self, dataset):
        '''
        Entrada: (Dataset) Dataset tokenizado

        Salva o Dataset no disco
        '''
        dataset.save_to_disk("dataset_encoded.hf")


    def setEmbeddingsOnModel(self, model_ckpt):
        '''
        Entrada: (Model) Modelo do Hugging faces
        Saída: (Model) Modelo oriundo do passado como parâmetro
        '''
        model = AutoModel.from_pretrained(model_ckpt).to(self.device)
        return model


    def getHiddenStatesOutputs(self, text, model):
        '''
        Entradas: (String) Texto para ser analisado
                  (Model) Modelo utilizado
        Saída:  (Tensor) Outputs da rede para o texto de entrada
        '''
        inputs = self.tokenizer(text, return_tensors = "pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states = True)
        return outputs


    def extract_hidden_states(self, batch):
        '''
        Entrada: (dict) Batch de dados com entradas do modelo
        Saída: (dict) Dicionários contendo um tensor
            referente aos pesos das camadas ocultas extraídas, 
            e suas respectivas labels

        Essa função extrai apenas para a última camada do modelo
        '''

        inputs = {k:v.to(self.device) for k,v in batch.items() 
                if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


    def extract_all_hidden_states(self, batch):
        '''
        Entrada: (dict) Batch de dados com entradas do modelo
        Saída: (dict) Dicionários contendo um tensor
            referente aos pesos das camadas ocultas extraídas, 
            e suas respectivas labels

        Essa função extrai para todas as camadas ocultas do modelo
        '''
        print("onde estamos?\n\n\n\n")
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
        Entrada: (Dataset) Dataset tokenizado
        Saída: (Dataset) Dataset formatado para PyTorch
        '''
        dataset_encoded.set_format("torch", 
                            columns=["input_ids", "attention_mask", "label"])
        return dataset_encoded


    def get_embeddings(self, X, y):
        '''
        Entrada: (array) Matriz de embddings para features X
                 (array) Vetor de embeddings para labels y
        Saída: (DataFrame) DataFrame com as embeddings 2D
        '''
        X_scaled = MinMaxScaler().fit_transform(X)
        mapper = UMAP(n_components=2, metric="cosine", random_state=42).fit(X_scaled)
        df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
        df_emb["label"] = y
        return df_emb


    def plot_map(self, dataset, hidden_state_label, map_dimension):
        '''
        Entrada: (Dataset) Dataset contendo as camadas ocultas
                 (string) Categorias das camadas ocultas a ser plotado
        Saída: Exibe um gráfico de dispersão 2D das embeddings
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
        Entrada: (Dataset) Dataset contendo todas as camadas ocultas
        Saída: Plota os mapas de todas as embeddings das camadas ocultas
        '''
        for hs in [x for x in dataset_all_hidden.column_names if x.startswith("hidden_state")]:
            print(hs)
            self.plot_map(dataset, hs, map_dimension)
    
   
    def hiddenStatesToNumpy(self, dataset_hidden):
        '''
        Entrada: (Dataset) Dataset contendo todas as camadas ocultas
        Saída:   (array, array) Arrays numpy com as embeddings das camadas
                ocultas e labels
        '''

        X = np.array(dataset_hidden["hidden_state"])
        y = np.array(dataset_hidden["label"])
        return X, y

    def get_grid(self, dataset, hidden_state_label, gridsize):
        '''
        Retornando o dataframe contendo os embeddings para cada 
        camada oculta da rede
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
        for hs in [x for x in dataset.column_names if x.startswith("hidden_state")]:
            print(hs)
            df_grid = self.get_grid(dataset, hs, gridsize)
            ct = pd.crosstab(df_grid.Y, df_grid.X, normalize=False)

            ct = ct.sort_index(ascending=False)
            
            sns.heatmap(ct, cmap="Blues", cbar=False, annot=True, fmt="d")
            #change figure title to hs
            plt.title(hs)
            plt.show()
