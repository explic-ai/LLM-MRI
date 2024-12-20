# LLM-MRI: a brain scanner for LLMs

As the everyday use of large language models (LLMs) expands, so does the necessity of understanding how these models achieve their designated outputs. While many approaches focus on the interpretability of LLMs through visualizing different attention mechanisms and methods that explain the model's architecture, `LLM-MRI` focuses on the activations of the feed-forward layers in a transformer-based LLM.

By adopting this approach, the library examines the neuron activations produced by the model for each distinct label. Through a series of steps, such as dimensionality reduction and representing each layer as a grid, the tool provides various visualization methods for the activation patterns in the feed-forward layers. Accordingly, the objective of this library is to contribute to LLM interpretability research, enabling users to explore visualization methods, such as heatmaps and graph representations of the hidden layers' activations in transformer-based LLMs.

This model allows users to explore questions such as:

- How do different categories of text in the corpus activate different neural regions?
- What are the differences between the properties of graphs formed by activations from two distinct categories?
- Are there regions of activation in the model more related to specific aspects of a category?

We encourage you to not only use this toolkit but also to extend it as you see fit.

## Index
- [Online Example](#online-example)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
  - [Activation Extraction](#activation-extraction)
  - [Heatmap Representation of Activations](#heatmap-representation-of-activations)
  - [Graph Representation of Activations](#graph-representation-of-activations)
  - [Composed Graph Visualization](#composed-graph-visualization)


## Online Example

The link below runs an online example of our library, in the Jupyter platform running over the Binder server:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/luizcelsojr/LLM-MRI/v01.2?labpath=examples%2FEmotions.ipynb)

## Instalation

To see LLM-MRI in action on your own data:

```
pip install llm_mri
```

## Usage

Firstly, the user needs to import the `LLM-MRI` and `matplotlib.pyplot` packages:

```
from llm_mri import LLM_MRI
import matplotlib.pyplot as plt
```
The user also needs to specify the Hugging Face Dataset that will be used to process the model's activations. There are two ways to do this:


- Load the Dataset from Hugging Face Hub: 
  ```
  dataset_url = "https://huggingface.co/datasets/dataset_link"
  dataset = load_dataset("csv", data_files=dataset_url)
  ```
- If you already have the dataset loaded on your machine, you can use the _load_from_disk_ function:
  ```
  dataset = load_from_disk(dataset_path) # Specify the Dataset's path
  ```
> Make sure that the selected dataset is a HuggingFace Dataset, and contains the columns "text" and "label", the last one being "ClassLabel" type. For more instructions on how to make this conversion, check out some of the examples on the [GitHub documentation.](https://github.com/explic-ai/LLM-MRI/tree/main/examples)


Next, the user selects the model to be used as a string:
```
model_ckpt = "distilbert/distilbert-base-multilingual-cased"
```
Then, the user instantiates `LLM-MRI`, to apply the methods defined on Functions:
```
llm_mri = LLM_MRI(model=model_ckpt, device="cpu", dataset=dataset)
```
## Functions
The library's functionality is divided into the following sections:

### Activation Extraction: 
As the user inputs the model and corpus to be analyzed, the dimensionality of the model's hidden layers is reduced, enabling visualization as an NxN grid.
  ```
  llm_mri.process_activation_areas(map_dimension)
  ```


  
### Heatmap representation of activations:
This includes the _get_layer_image_ function, which transforms the NxN grid for a selected layer into a heatmap. In this heatmap, each cell represents the number of activations that different regions on a determined layer received for the provided corpus. Additionally, users can visualize activations for a specific label.
  ```
  fig = llm_mri.get_layer_image(layer, category)
  ```
![hidden_state_1_true](https://github.com/user-attachments/assets/0bfbc90e-2bb9-4bd0-aa20-68c67608189f)



  
### Graph Representation of Activations:
Using the _get_graph_ function, the module connects regions from neighboring layers based on co-activations to form a graph representing the entire network. The graph's edges can also be colored according to different labels, allowing the user to identify the specific category that activated each neighboring node.

> **_colormap:_**  The default used colormap is the 'coolwarm'. More can be found on [matplotlib.colors](https://matplotlib.org/stable/users/explain/colors/colormaps.html). We recommend the use of a 'Diverging' colormap for better visualization.
> **_fix_node_positions:_**  'True' keeps the nodes and edges at the same positions, independently of the categories. This could be useful for comparing activations between distinct categories. Setting to 'False' does not allow this comparison, although the graph will be more easily visualized.

   ```
   graph = llm_mri.get_graph(category)
   graph_image = llm_mri.get_graph_image(graph, colormap, fix_node_positions)
  ```
![true](https://github.com/user-attachments/assets/98b006ad-1e1a-40c1-9259-66e0496203b8)



### Composed Graph Visualization:
The user is also able to obtain a composed visualization of two different categories using the _get_composed_graph_ function. By setting a category, each edge is colored based on the designated label, so the user is able to see which document label activated each region. Additionally, the user can select a colormap, where node colors reflect the label that most strongly activated them. Nodes colored white indicate equal activation by both categories, as white represents the midpoint of the color spectrum.
```
g_composed = llm_mri.get_composed_graph("true", "fake")
g_composed_img = llm_mri.get_graph_image(g_composed)
```

![fake_and_true_graph](https://github.com/user-attachments/assets/7ca1c194-045f-45fd-a2a7-33941fe0dc86)




