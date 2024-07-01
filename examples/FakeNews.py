import sys
sys.path.insert(1, '../visualization/src')
from Treatment import Treatment
from LLM_MRI import LLM_MRI

model_ckpt = "distilbert/distilbert-base-multilingual-cased"
handle = Treatment(model=model_ckpt, device="cpu")

# Turning Pandas DataFrame into HuggingFace Dataset (in case your Dataset is not HF already)
df = handle.le_documentos("/home/lipecorradini/desktop/unicamp/ic/vizactv/fake-br-corpus-sample")
stopwords = handle.set_stopwords() # Defining stopwords

handle.set_normalizado(df, stopwords) # Text Normalization

dataset = handle.DfToHuggingFacesDataset(df, class_names=["true", "fake"]) # Transformando df em um Dataset do HuggingFace

# Beginning Visualization

my_viz = LLM_MRI(model=model_ckpt, device="cpu", dataset=dataset)

my_viz.setDataset(my_viz.initialize_dataset()) # Initializing Encoded Dataset

my_viz.process_activation_areas(map_dimension = 10) # Getting activation Areas and Reducing Dimensionality

my_viz.get_layer_image(layer = 1, category=0) # Getting the image for a specific layer and specific label category (Ex: label = 0)

my_viz.get_graph_image(category=0) # Getting the graph image for a determined category

my_viz.get_graph(category=0) # Getting the graph for a designed category

g = my_viz.get_all_graph() # Getting the image of the whole graph

