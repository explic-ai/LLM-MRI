import torch
import matplotlib.pyplot as plt

dimensions = 100

def svd(tensor_data):
    # Create a random tensor of shape (40, 768)

    # Perform SVD
    U, s, Vt = torch.linalg.svd(tensor_data, full_matrices=False)

    # Select the top 100 components
    U_k = U[:, :dimensions]  # Take the first dimensions columns of U
    s_k = s[:dimensions]     # Take the first dimensions singular values
    Vt_k = Vt[:dimensions, :dimensions]  # Take the first 100 rows of Vt

    print("U shape:", U_k.shape)
    print("s shape:", s_k.shape)
    print("Vt shape:", Vt_k.shape)

    # Reconstruct the reduced tensor
    reduced_tensor = U_k @ torch.diag(s_k) @ Vt_k

    print("Original shape:", tensor_data.shape)
    print("Reduced shape:", reduced_tensor.shape)

    return reduced_tensor


def render_heatmap(tensor_data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Convert tensor to a Pandas DataFrame for easier manipulation
    df = pd.DataFrame(tensor_data.numpy())

    df = df.T
    # Compute the Spearman correlation matrix
    spearman_corr = df.corr(method='spearman')

    # Plot the heatmap using seaborn
    plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
    sns.heatmap(spearman_corr, cmap='coolwarm', annot=False, fmt=".2f")


def spearman_correlation(first_layer, second_layer):
    """
    Compute Spearman correlation between the components of two different layers.
    Args: 
        first_layer (tensor): the first layer to be used in the correlation
        second_layer (tensor): the second layer to be used in the correlation
    """
    # Rank the columns of each tensor
    rank1 = first_layer.argsort(dim=0).argsort(dim=0).float()
    rank2 = second_layer.argsort(dim=0).argsort(dim=0).float()

    # Center the ranks
    rank1 -= rank1.mean(dim=0, keepdim=True)
    rank2 -= rank2.mean(dim=0, keepdim=True)

    # Compute the covariance and standard deviations
    cov = (rank1.T @ rank2) / first_layer.size(0)
    std1 = rank1.std(dim=0, keepdim=True)
    std2 = rank2.std(dim=0, keepdim=True)

    # Compute the correlation matrix
    correlation_matrix = cov / (std1.T @ std2)
    
    return correlation_matrix

def render_spearman_heatmap(correlation_matrix):
    """
    Renders a heatmap from a Spearman correlation matrix.

    Args:
        correlation_matrix (torch.Tensor or np.ndarray): The Spearman correlation matrix.
        title (str): Title of the heatmap.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Convert the correlation matrix to a NumPy array if it's a PyTorch tensor
    if isinstance(correlation_matrix, torch.Tensor):
        correlation_matrix = correlation_matrix.numpy()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f")

if __name__ == "__main__":
    # Example usage
    # svd()
    tensor_data_5 = torch.load("./examples/full_hs_5.pt")
    print(tensor_data_5.shape)

    tensor_data_6 = torch.load("./examples/full_hs_6.pt")
    print(tensor_data_6.shape)

    correlation_full = spearman_correlation(tensor_data_5, tensor_data_6)
    correlation_reduced = spearman_correlation(svd(tensor_data_5), svd(tensor_data_6))

    render_spearman_heatmap(correlation_full)
    plt.title("Mapa de calor da correlação de Spearman dos exemplos da matriz original")

    
    render_spearman_heatmap(correlation_reduced)
    plt.title(f"Mapa de calor da correlação de Spearman da matriz reduzida com {dimensions} componentes")

    plt.show()

    # Agora, vamos comparar a matriz de correlação entre as componentes antes e depois da reconstrução