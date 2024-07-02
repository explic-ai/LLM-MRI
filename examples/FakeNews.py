import sys
sys.path.insert(1, '../')
import Treatment 
import LLM_MRI

model_ckpt = "distilbert/distilbert-base-multilingual-cased"
handle = Treatment(model=model_ckpt, device="cpu")

# Turning Pandas DataFrame into HuggingFace Dataset (in case your Dataset is not HF already)
df = handle.le_documentos("/home/lipecorradini/desktop/unicamp/ic/vizactv/fake-br-corpus-sample")
stopwords = handle.set_stopwords() # Defining stopwords

handle.set_normalizado(df, stopwords) # Text Normalization

dataset = handle.DfToHuggingFacesDataset(df, class_names=["true", "fake"]) # Transformando df em um Dataset do HuggingFace

# Beginning Visualization

llm_mri = LLM_MRI(model=model_ckpt, device="cpu", dataset=dataset)

llm_mri.get_layer_image(layer = 1, category=0) # Getting the image for a specific layer and specific label category (Ex: label = 0)

llm_mri.get_graph_image(category=0) # Getting the graph image for a determined category

g1 = llm_mri.get_graph(category=0) # Getting the graph for a designed category

g = llm_mri.get_all_graph() # Getting the image of the whole graph

llm_mri.get_all_graph_image()

# hidden_states_dataset = 
llm_mri.process_activation_areas(map_dimension = 10) # Getting activation Areas and Reducing Dimensionality