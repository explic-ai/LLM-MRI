import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

class Metrics:

    def __init__(self, Graph, label, model_name): # Revisar isso daqui
        self.Graph = Graph
        self.label = label
        self.model_name = model_name
        self.layers = Graph.graph['layers']

        """
        Rotulando cada nó com a camada que ele pertence, isso é feito
        buscando o primeiro número do seu nome
        """
        for n in self.Graph.nodes:
            self.Graph.nodes[n]['layer'] = int(n.split('_')[0])
        
        self.projection_even, self.projection_odd = self.project_graph()

    def project_graph(self):
        nodes_even_layers = set()
        nodes_odd_layers = set()
        """
        Para fazer a projeção, é necessário que os nós tenham a label
        de qual camada pertencem.
        """
        for layer in range(self.layers + 1):
            if layer % 2 == 0:
                nodes_even_layers = nodes_even_layers.union({ n for n, d in self.Graph.nodes(data=True) if d['layer'] == layer})
            else:
                nodes_odd_layers = nodes_odd_layers.union({ n for n, d in self.Graph.nodes(data=True) if d['layer'] == layer})

        return bipartite.collaboration_weighted_projected_graph(self.Graph, nodes_even_layers), bipartite.collaboration_weighted_projected_graph(self.Graph, nodes_odd_layers)

    def get_degree_by_layer(self):
        camadas = []
        for x in range(self.layers + 1):
            camadas.append(str(x))
        df_layers = pd.DataFrame(columns=['layer', 'mean', 'var'])

        for i in camadas:
            df_layers = pd.concat([pd.DataFrame([[
                i,
                pd.Series([v for k, v in dict(nx.degree(self.Graph)).items() if k.split("_")[0] == i]).mean(),
                pd.Series([v for k, v in dict(nx.degree(self.Graph)).items() if k.split("_")[0] == i]).var(),
            ]], columns=df_layers.columns), df_layers], ignore_index=True)
        
        return df_layers.reindex(index=df_layers.index[::-1])

    def get_graph_center_of_mass(self):
        camadas = []
        for x in range(self.layers + 1):
            camadas.append(str(x))
        
        center_of_mass = 0

        for i in camadas:
            center_of_mass += ((pd.Series([k for k, v in dict(self.Graph.nodes()).items() if k.split("_")[0] == i]).count()) * (int(i) - (self.layers / 2)))
        
        return center_of_mass / len(list(self.Graph.nodes()))

    def get_graph_center_of_strength(self):
        camadas = []
        for x in range(self.layers + 1):
            camadas.append(str(x))
        
        center_of_strength = 0

        for i in camadas:
            center_of_strength += ((pd.Series([v for k, v in dict(self.Graph.degree(weight='weight')).items() if k.split("_")[0] == i]).std()) * (int(i) - (self.layers / 2)))
        
        return center_of_strength

    def get_graph(self):
        return self.Graph
    
    def get_basic_metrics(self):
        return {
            "mean_degree": pd.Series([v for k, v in dict(nx.degree(self.Graph)).items()]).mean(),
            "var_degree": pd.Series([v for k, v in dict(nx.degree(self.Graph)).items()]).var(),
            "skew_degree": pd.Series([v for k, v in dict(nx.degree(self.Graph)).items()]).skew(),
            "kurt_degree": pd.Series([v for k, v in dict(nx.degree(self.Graph)).items()]).kurt(),
            "mean_strength": pd.Series([v for k, v in dict(self.Graph.degree(weight='weight')).items()]).mean(),
            "var_strength": pd.Series([v for k, v in dict(self.Graph.degree(weight='weight')).items()]).var(),
            "skew_strength": pd.Series([v for k, v in dict(self.Graph.degree(weight='weight')).items()]).skew(),
            "kurt_strength": pd.Series([v for k, v in dict(self.Graph.degree(weight='weight')).items()]).kurt(),
            # "average_node_connectivity": nx.average_node_connectivity(self.Graph),
            "assortativity": nx.degree_assortativity_coefficient(self.Graph, weight='weight'),
            "density": nx.density(self.Graph),
            "center_of_mass": self.get_graph_center_of_mass(),
            "center_of_strength": self.get_graph_center_of_strength(),
            "model_name": self.model_name,
            "label": self.label
        }

    def get_projection_metrics_even(self):
        return {
            "mean_degree": pd.Series([v for k, v in dict(nx.degree(self.projection_even)).items()]).mean(),
            "var_degree": pd.Series([v for k, v in dict(nx.degree(self.projection_even)).items()]).var(),
            "mean_strength": pd.Series([v for k, v in dict(self.projection_even.degree(weight='weight')).items()]).mean(),
            "var_strength": pd.Series([v for k, v in dict(self.projection_even.degree(weight='weight')).items()]).var(),
            "average_clustering": nx.average_clustering(self.projection_even, weight="weight"),
            # "average_node_connectivity": nx.average_node_connectivity(self.projection_even),
            "assortativity": nx.degree_assortativity_coefficient(self.projection_even, weight='weight'),
            "density": nx.density(self.projection_even),
            "average_shortest_path": nx.average_shortest_path_length(self.projection_even, weight="weight") if nx.is_connected(self.projection_even) else float('NaN'),
            "model_name": self.model_name,
            "label": self.label,
            "side": "even"
        }

    def get_projection_metrics_odd(self):
        return {
            "mean_degree": pd.Series([v for k, v in dict(nx.degree(self.projection_odd)).items()]).mean(),
            "var_degree": pd.Series([v for k, v in dict(nx.degree(self.projection_odd)).items()]).var(),
            "mean_strength": pd.Series([v for k, v in dict(self.projection_odd.degree(weight='weight')).items()]).mean(),
            "var_strength": pd.Series([v for k, v in dict(self.projection_odd.degree(weight='weight')).items()]).var(),
            "average_clustering": nx.average_clustering(self.projection_odd, weight="weight"),
            # "average_node_connectivity": nx.average_node_connectivity(self.projection_odd),
            "assortativity": nx.degree_assortativity_coefficient(self.projection_odd, weight="weight"),
            "density": nx.density(self.projection_odd),
            "average_shortest_path": nx.average_shortest_path_length(self.projection_odd, weight="weight") if nx.is_connected(self.projection_odd) else float('NaN'),
            "model_name": self.model_name,
            "label": self.label,
            "side": "odd"
        }

    def get_basic_metrics_list_of_names(self):
        return [
            'mean_degree',
            'var_degree',
            'skew_degree',
            'kurt_degree',
            "mean_strength",
            "var_strength",
            "skew_strength",
            "kurt_strength",
            'average_clustering',
            'assortativity',
            'density',
            'model_name',
            'label',
            'side',
        ]