from visualization.src.Treatment import Treatment

model_ckpt = "distilbert/distilbert-base-multilingual-cased"
my_viz = Treatment(model=model_ckpt, device="cpu")

# Testando função para ler documentos
df = my_viz.le_documentos("/home/lipecorradini/desktop/unicamp/ic/vizactv/fake-br-corpus-sample")
print(my_viz.head())

# Testando Função para definir as stopwords
stopwords = my_viz.set_stopwords()

# Testar a normalização do texto
my_viz.set_normalizado(df, stopwords)
print(my_viz.columns) # Nome das colunas terá que ser padronizado

# Transformando df em um Dataset do HuggingFace
dataset = my_viz.DfToHuggingFacesDataset(df, class_names=["true", "fake"])
print(dataset)

# Tokenizando a entrada
print(my_viz.tokenize(dataset[:2]))

# Criando o Dataset a partir dos DataFrames utilizados para treinamento e teste
encodedDataset = my_viz.encodeDataset(dataset)

# Salvando o Dataset no disco
my_viz.saveDataset(encodedDataset)

# Selecionando o modelo que será utilizado a partir do pré-treinado model_ckpt
model = my_viz.setEmbeddingsOnModel(model_ckpt)

# Obtendo os outputs das camadas ocultas da rede
outputs = my_viz.getHiddenStatesOutputs("this is a test", model)
print(outputs)

# Transformando o dataset em torch
encodedDataset = my_viz.setDatasetToTorch(encodedDataset)

# Obtendo as camadas ocultas do dataset
datasetAllHiddenStates = encodedDataset.map(my_viz.extract_all_hidden_states, batched=True)
print(datasetAllHiddenStates.column_names)

# Obtendo os Embeddings de camadas ocultas
datasetHidden = encodedDataset.map(my_viz.extract_hidden_states, batched=True)
print(datasetHidden[:5])
X, y = my_viz.hiddenStatesToNumpy(datasetHidden)
EmbeddingsDf = my_viz.get_embeddings(X, y)
print(EmbeddingsDf[:5])

# Plotando as visualizações
my_viz.plot_maps_all_hidden(datasetAllHiddenStates)







