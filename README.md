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

## Functions
The library's functionality is divided into the following sections:

**Activation Extraction:** As the user inputs the model and corpus to be analyzed, the dimensionality of the model's hidden layers is reduced, enabling visualization as an NxN grid.
  ```
  llm_mri.process_activation_areas(map_dimension)
  ```

  
**Heatmap representation of activations:** This includes the _get_layer_image_ function, which transforms the NxN grid for a selected layer into a heatmap. In this heatmap, each cell represents the number of activations that different regions received for the provided corpus. Additionally, users can visualize activations for a specific label.
  ```
  fig = llm_mri.get_layer_image(layer, category)
  ```
![hidden_state_1_true](https://github.com/user-attachments/assets/0bfbc90e-2bb9-4bd0-aa20-68c67608189f)

  
**Graph Representation of Activations**: Using the _get_graph_ function, the module connects regions from neighboring layers based on co-activations to form a graph representing the entire network. The graph's edges can also be colored according to different labels, allowing the user to identify the specific category that activated each neighboring node.
   ```
   graph = llm_mri.get_graph(category)
   graph_image = llm_mri.get_graph_image(graph)
  ```
![Captura de tela de 2024-07-15 13-24-28](https://github.com/user-attachments/assets/327b8c94-1162-4e2b-8b1b-d1be2fb1163e)



