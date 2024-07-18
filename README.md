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
- [Execution](#execution)
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

**Clone this repository on your machine**

```
git clone https://github.com/luizcelsojr/LLM-MRI

cd LLM-MRI
```

**Install Poetry**

```
pip install poetry
```


**Install Graphviz**

```
sudo apt install graphviz

sudo apt install libgraphviz-dev
```

**Install other dependencies**

```
poetry install --no-root
```

## Execution

**Enable poetry's shell:**
```
poetry shell
```

To run your python file:
```
python3 file.py
```

To run your jupyter notebook:
```
poetry run jupyter notebook
```
## Usage

Firstly, the user needs to import the `LLM-MRI` and `matplotlib,pyplot` packages:

```
import LLM_MRI
import matplotlib.pyplot as plt
```
The user also needs to specify the Hugging Face Dataset that will be used to process the model's activations. There are two ways to do this:


- Load the Dataset from Hugging Face Hub: 
  ```
  dataset_url = "https://huggingface.co/datasets/dataset_link"
  dataset = load_dataset("csv", data_files=dataset_url)
  ```
- If you already has the dataset loaded on your machine, you can use the _load_from_disk_ function:
  ```
  dataset = load_from_disk(dataset_path) # Specify the Dataset's path
  ```

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
This includes the _get_layer_image_ function, which transforms the NxN grid for a selected layer into a heatmap. In this heatmap, each cell represents the number of activations that different regions received for the provided corpus. Additionally, users can visualize activations for a specific label.
  ```
  fig = llm_mri.get_layer_image(layer, category)
  ```
![hidden_state_1_true](https://github.com/user-attachments/assets/0bfbc90e-2bb9-4bd0-aa20-68c67608189f)



  
### Graph Representation of Activations:
Using the _get_graph_ function, the module connects regions from neighboring layers based on co-activations to form a graph representing the entire network. The graph's edges can also be colored according to different labels, allowing the user to identify the specific category that activated each neighboring node.
   ```
   graph = llm_mri.get_graph(category)
   graph_image = llm_mri.get_graph_image(graph)
  ```
![Captura de tela de 2024-07-15 13-24-28](https://github.com/user-attachments/assets/327b8c94-1162-4e2b-8b1b-d1be2fb1163e)


The user is also able to obtain a composed visualization of two different categories using the _get_composed_graph_ function. By setting a category, each edge is colored based on the designated label, so the user is able to see which document label activated each region.
```
g_composed = llm_mri.get_composed_graph("true", "fake")
g_composed_img = llm_mri.get_graph_image(g_composed)
```

![new_colored_graph(2)](https://github.com/user-attachments/assets/05fee9a7-a3e3-4e67-92f8-d60175de6110)

