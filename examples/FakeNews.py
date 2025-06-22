import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from llm_mri import LLM_MRI
import matplotlib.pyplot as plt
from datasets import load_from_disk

model_ckpt = "distilbert/distilbert-base-multilingual-cased"
# The model can also be an encoder, such as 'openai-community/gpt2'

dataset_folder = os.path.join(os.path.dirname(__file__), '..', 'dataset')
print(dataset_folder)
dataset_path = os.path.join(dataset_folder, 'dataset_encoded.hf')

# Load the dataset using Hugging Face's `load_dataset` function
dataset = load_from_disk(dataset_path)
dataset.cleanup_cache_files()

print(dataset)
print(dataset[0])

# Beginning Visualization
llm_mri = LLM_MRI(model=model_ckpt, device="cpu", dataset=dataset)

dimensions = 30

# Processing hidden states and activation areas
llm_mri.process_activation_areas(map_dimension = dimensions) # Getting activation Areas and Reducing Dimensionality, as a torch dataset

# Getting the layer's image for a designed category
# fig = llm_mri.get_layer_image(layer = 1, category="fake") # Getting the image for a specific layer and specific label category (Ex: label = 0)
# plt.tight_layout()
# plt.show()

# Getting full scatterplot
# fig_scatter = llm_mri.get_original_map(6)
# plt.tight_layout()
# plt.show()

# fig = llm_mri.get_layer_image(layer = 1, category="true") # Getting the image for a specific layer and specific label category (Ex: label = 0)
# plt.tight_layout()
# plt.show()

# g = llm_mri.get_graph()
# _ = llm_mri.get_graph_image(g)
# plt.show()

# Getting activation's image as a Graph
# g = llm_mri.get_graph(category_name="true") # Getting the graph for a designed category

g_full = llm_mri.get_svd_graph() # Gets the graph for all categories

# Getting the image of Graph representation of activations
g_img = llm_mri.get_graph_image(g_full, fix_node_positions=True, fix_node_dimensions=False) # Getting the graph image for a determined category
plt.title("Dimensionality Reduction of full graph by PCA")

plt.box(False)
plt.show()

# Getting activations of different labels in the same Graph
g_composed = llm_mri.get_composed_graph("true", "fake")

# Generating image of composed graph
g_composed_img = llm_mri.get_graph_image(g_composed)  # default: coolwarm
plt.title("Full graph using UMAP as dimensionality reduction")
plt.box(False)
# plt.show()

# print("Reduzindo por PCA com 50 componentes:")
# Generating image of svd composed graph
# svd_composed = llm_mri.get_composed_svd_graph("true", "fake", dim=50)

# svd_full_img = llm_mri.get_graph_image(svd_composed, fix_node_positions=True, fix_node_dimensions=True)
# plt.title("Dimensionality Reduction to 50 dimensions of distinct categories by PCA")

svd_composed = llm_mri.get_composed_svd_graph("fake", "true", threshold=0.5)

svd_full_img = llm_mri.get_graph_image(svd_composed, fix_node_positions=True, fix_node_dimensions=True)
plt.title(f"Dimensionality Reduction to {dimensions} dimensions of distinct categories by PCA")
# plt.box(False)
plt.show()
