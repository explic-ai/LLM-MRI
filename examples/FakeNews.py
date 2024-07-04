import sys
sys.path.insert(1, '../')

import LLM_MRI
import matplotlib.pyplot as plt
import os
from datasets import load_from_disk

model_ckpt = "distilbert/distilbert-base-multilingual-cased"

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'dataset')
dataset_path = os.path.join(dataset_folder, 'dataset_encoded.hf')

# Load the dataset using Hugging Face's `load_dataset` function
dataset = load_from_disk(dataset_path)

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

