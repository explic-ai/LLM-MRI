from llm_mri import LLM_MRI
import matplotlib.pyplot as plt
from datasets import load_from_disk
import os

model_ckpt = "distilbert/distilbert-base-multilingual-cased"

dataset_path = "./datasets/animals_encoded.hf"

# Load the dataset using Hugging Face's `load_dataset` function
dataset = load_from_disk(dataset_path)
dataset.cleanup_cache_files()

dataset = dataset.remove_columns(['Unnamed: 0','name','__index_level_0__'])

print(dataset)
print(dataset[0])

def find_bad_rows(dataset):
    bad_indices = []
    for i in range(len(dataset)):
        try:
            # Test if tokenizer can process this text
            _ = llm_mri.base.tokenizer(dataset[i]["text"])
        except:
            bad_indices.append(i)
    return bad_indices

bad_rows = find_bad_rows(dataset)
print(f"Problematic rows: {bad_rows}")

llm_mri = LLM_MRI(model=model_ckpt, device="cpu", dataset=dataset)

dimensions = 20

# Processing hidden states and activation areas
llm_mri.process_activation_areas(map_dimension = dimensions) # Getting activation Areas and Reducing Dimensionality, as a torch dataset

svd_composed = llm_mri.get_composed_svd_graph("bird", "fish", threshold=0.5)

svd_full_img = llm_mri.get_graph_image(svd_composed, fix_node_positions=True, fix_node_dimensions=True)
plt.title(f"Dimensionality Reduction to {dimensions} dimensions of distinct categories by PCA")
# plt.box(False)
plt.show()