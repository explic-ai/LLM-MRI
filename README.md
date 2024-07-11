# LLM-MRI: a brain scanner for LLMs

As the everyday use of large language models (LLMs) grows, so does the necessity of understanding how LLMs achieve their designed outputs.

One approach to analyzing LLM interpretability is to examine the neuron activations produced by the model for each different label. Accordingly, the objective of this library is to contribute to LLM interpretability research, as it allows users to explore various types of activation visualization methods, such as heatmaps and graph representation, to ensure the model's interpretability.

We encourage you to not only use this toolkit but also to extend it as you see fit.

## Online Example

The link below runs an online example of our library, in the Jupyter platform running over the Binder server:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/luizcelsojr/LLM-MRI/v01.1?labpath=examples%2FEmotions.ipynb)

## Instalation

To see LLM-MRI in action on your own data:

**Clone this repository on your machine**

```
git clone https://github.com/luizcelsojr/LLM-MRI
```

**Install Poetry**

```
pipx install poetry

```


**Install Graphviz**

```
sudo apt install graphviz

```
```
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

## Functions
The workings of the library is divided in:

- **Activation Extraction:** As the user passes the model and corpus to be analyzed, reduces the model's hidden layers dimensionality, so that it can be visualized as a NxN grid.
  ```
  llm_mri.process_activation_areas(map_dimension)
  ```
  
- **Heatmap representation of activations:** Contains the _get_layer_image_ function, which turns NxN grid for a chosen layer into a heatmap, so that a cell represents the amount of activations that each regions got for the passed corpus. The user is also able to visualize the activations for a specific category.
  ```
  fig = llm_mri.get_layer_image(layer, category)
  ```
  
- **Graph Representation of Activations**: Through the _get_graph_ function, the module connects regions from neighbor layers, based on co-activations, to form a graph representing the entire network. The graph's edges can also be colored based on different labels, so that the user is able to verify the specific category that activated each neighbor nodes.
   ```
   graph = llm_mri.get_graph(category_name)
   graph_image = llm_mri.get_graph_image(graph)
  ```
