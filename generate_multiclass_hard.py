import numpy as np

def generate_complex_libsvm_dataset(K, m, n, sparsity=0.1, filename='complex_dataset.txt'):
    """
    Generates a more complex dataset in LIBSVM format with specified characteristics
    to make classification tasks less smooth and more challenging.
    
    Args:
    K (int): Number of classes.
    m (int): Total number of observations.
    n (int): Number of features.
    sparsity (float): Fraction of features set as zero (sparsity).
    filename (str): Path and name of the file to save the dataset.
    
    Returns:
    None: Saves the dataset to a text file in LIBSVM format.
    """
    base_count = m // K
    min_count = int(base_count * 0.8)
    max_count = int(base_count * 1.2)
    
    counts = np.random.randint(min_count, max_count + 1, size=K)
    while sum(counts) != m:
        counts = np.random.randint(min_count, max_count + 1, size=K)
        if sum(counts) > m:
            counts[np.argmax(counts)] -= sum(counts) - m
        elif sum(counts) < m:
            counts[np.argmin(counts)] += m - sum(counts)
    
    # Generate feature means and covariances dynamically for complexity
    class_means = [np.random.randn(n) * 5 for _ in range(K)]  # Multiply by factor to increase mean distance
    class_covariances = [(np.diag(np.random.rand(n)) + 0.5) * np.random.rand(1) * 10 for _ in range(K)]  # Diverse covariance matrices

    with open(filename, 'w') as file:
        for i in range(K):
            # Generate data for each class with specified covariance
            class_features = np.random.multivariate_normal(mean=class_means[i], cov=class_covariances[i], size=counts[i])
            # Introduce sparsity
            mask = np.random.rand(counts[i], n) < sparsity
            class_features[mask] = 0
            
            # Write each observation to the file
            for features in class_features:
                line = str(i)
                for index, value in enumerate(features, start=1):
                    if value != 0:
                        line += f" {index}:{value:.4f}"
                line += "\n"
                file.write(line)

    print(f'Complex dataset with {m} observations, {n} features, and {K} classes saved to {filename}')

# Example usage
generate_complex_libsvm_dataset(K=3, m=2000, n=16, sparsity=0.2, filename='/Users/shaozishan/Desktop/Research/24SpringResearch/ca_reg_path/synthetic_multiclass_hard.txt')
