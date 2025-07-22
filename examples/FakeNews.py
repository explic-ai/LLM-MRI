import os

from llm_mri import LLM_MRI
from llm_mri.dimensionality_reduction import PCA

import matplotlib.pyplot as plt
from datasets import load_from_disk

model_ckpt = "distilbert/distilbert-base-multilingual-cased"
# The model can also be an encoder, such as 'openai-community/gpt2'

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'dataset')
dataset_path = os.path.join(dataset_folder, 'dataset_encoded.hf')

# Load the dataset using Hugging Face's `load_dataset` function
dataset = load_from_disk(dataset_path)
dataset.cleanup_cache_files()

# Defining the dimensionality reduction method
pca = PCA(n_components = 20)

# Beginning Visualization
llm_mri = LLM_MRI(model=model_ckpt, device="cpu", dataset=dataset, reduction_method=pca)

# Processing hidden states and activation areas
llm_mri.process_activation_areas() # Getting activation Areas and Reducing Dimensionality, as a torch dataset

g_full = llm_mri.get_graph("true") # Gets the graph for the true category
g_img = llm_mri.get_graph_image(g_full, fix_node_dimensions=False)
plt.title("Dimensionality Reduction of true graph by PCA")

g_full = llm_mri.get_graph("fake") # Gets the graph for the fake category
g_img = llm_mri.get_graph_image(g_full, fix_node_dimensions=True)
plt.title("Dimensionality Reduction of fake graph by PCA")

g_full = llm_mri.get_graph(["true", "fake"]) # Gets the graph for all categories
g_img = llm_mri.get_graph_image(g_full, fix_node_dimensions=True)
plt.title("Dimensionality Reduction of fake and true graph by PCA")

plt.box(False)
plt.show()