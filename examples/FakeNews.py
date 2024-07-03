import sys
sys.path.insert(1, '../')

import Treatment 
import LLM_MRI
import matplotlib.pyplot as plt

model_ckpt = "distilbert/distilbert-base-multilingual-cased"
handle = Treatment(model=model_ckpt, device="cpu")

# Turning Pandas DataFrame into HuggingFace Dataset (in case your Dataset is not HF already)
df = handle.le_documentos("/home/lipecorradini/desktop/unicamp/ic/vizactv/fake-br-corpus-sample")
stopwords = handle.set_stopwords() # Defining stopwords

handle.set_normalizado(df, stopwords) # Text Normalization

dataset = handle.DfToHuggingFacesDataset(df, class_names=["true", "fake"]) # Transformando df em um Dataset do HuggingFace



# Beginning Visualization
llm_mri = LLM_MRI(model=model_ckpt, device="cpu", dataset=dataset)

# Processing hidden states and activation areas
llm_mri.process_activation_areas(map_dimension = 10) # Getting activation Areas and Reducing Dimensionality, as a torch dataset

# Getting the layer's image for a designed category
fig = llm_mri.get_layer_image(layer = 1, category=0) # Getting the image for a specific layer and specific label category (Ex: label = 0)
plt.tight_layout()
plt.show()

# Getting activation's image as a Graph
g = llm_mri.get_graph(category=0) # Getting the graph for a designed category

# Getting the image of Graph representation of activations
g_img = llm_mri.get_graph_image(category=0) # Getting the graph image for a determined category
plt.box(False)
plt.show()

