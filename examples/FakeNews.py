import sys
sys.path.insert(1, '../src')

import LLM_MRI
import matplotlib.pyplot as plt
import os
from datasets import load_from_disk
import networkx as nx

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
fig = llm_mri.get_layer_image(layer = 0, category="fake") # Getting the image for a specific layer and specific label category (Ex: label = 0)
plt.tight_layout()
plt.show()

fig = llm_mri.get_layer_image(layer = 0, category="true") # Getting the image for a specific layer and specific label category (Ex: label = 0)
plt.tight_layout()
plt.show()

# Getting activation's image as a Graph
g = llm_mri.get_graph(category_name="true") # Getting the graph for a designed category
g_full = llm_mri.get_graph() # Gets the graph for all categorys

# Getting the image of Graph representation of activations
g_img = llm_mri.get_graph_image(g) # Getting the graph image for a determined category
plt.box(False)
plt.show()

# Getting activations of different times in a same Graphs
g_false = llm_mri.get_graph(category_name="true")
g_true = llm_mri.get_graph(category_name="fake")
g_composed = nx.compose(g_true, g_false)

# Marking repeated edges
duplicates = list(set(g_true.edges) & set(g_false.edges))
for e in duplicates : g_composed.edges[e]['label'] = 2 

# Generating image of composed graph
g_composed_img = llm_mri.get_graph_image(g_composed)
plt.box(False)
plt.show()
