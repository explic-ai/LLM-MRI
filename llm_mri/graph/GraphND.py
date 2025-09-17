from ._base import Graph

class GraphND(Graph):

    def __init__(self, n_components:int):
        """
        Initialize the GraphND object.

        :param n_components: The number of components that the graph was reduced based on.
        """
        super().__init__(n_components)
        self.n_components = n_components

    
    def build_graph(self, dataset):
        if isinstance(categories, str):
            categories = [categories]
        
        if len(categories) > 2:
            raise ValueError("This method can only generate a graph for one or two categories. If you wish to compare more categories, please input one at a time")

        # List to store graphs for each category
        graphs_list = []

        for category in categories:

            if category not in self.class_names:
                raise ValueError(f"Category '{category}' is not in the dataset's class names.")
            
            # Get the category index    
            category_index = self.class_names.index(category)    

            # Filter the dataset to get the indices of rows with the given category
            indices = [i for i, label in enumerate(self.dataset['label']) if label == category_index]
            
            #  Extract the rows from the hidden_states_dataset tensor
            filtered_hidden_states = self.hidden_states_dataset.select(indices) 
        
            #  Select only rows with selected categories from hidden state
            full_svd_hs = self.reduction_method.get_reduction(filtered_hidden_states)

            # 2) Select specific hidden states to compute spearman correlation
            c_hidden_states = {}

            for i in range(len(full_svd_hs)):
                c_hidden_states[f'hidden_state_{i}'] = full_svd_hs[f'hidden_state_{i}']
            
            # Updating graphs list
            graph = self._get_spearman_graph(c_hidden_states, category_index, threshold)
            graphs_list.append(graph)
            
            # Updating the hidden states for the given category
            self.category_hidden_states[category] = c_hidden_states

        # Merging graphs (if more than one category)
        if len(graphs_list) > 1:
            G = nx.compose(graphs_list[0], graphs_list[1])
            G.graph['label_names'] = [graphs_list[0].graph['label_names'][0], graphs_list[1].graph['label_names'][0]]
        
        else:
            G = graphs_list[0]

        # Defining the number of layers on the graph's properties
        G.graph['layers'] = self.num_layers

        return G