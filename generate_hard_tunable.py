import numpy as np

def generate_customizable_libsvm_dataset(K, m, n, sparsity=0.1, mean_multiplier=5, cov_multiplier=10, filename='customizable_dataset.txt'):
    """
    Generates a customizable dataset in LIBSVM format with specified characteristics
    to make classification tasks adjustable in complexity.
    
    Args:
    K (int): Number of classes.
    m (int): Total number of observations.
    n (int): Number of features.
    sparsity (float): Fraction of features set as zero (sparsity).
    mean_multiplier (float): Multiplier for the separation of class means (higher values increase separability).
    cov_multiplier (float): Multiplier for the influence of covariance in data generation (higher values increase feature variance and complexity).
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
    
    # Generate feature means and covariances with adjustable complexity
    class_means = [np.random.randn(n) * mean_multiplier for _ in range(K)]
    class_covariances = [(np.diag(np.random.rand(n)) + 0.5) * np.random.rand(1) * cov_multiplier for _ in range(K)]

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

    print(f'Customizable dataset with {m} observations, {n} features, and {K} classes saved to {filename}')

# Example usage
generate_customizable_libsvm_dataset(K=3, m=1000, n=10, sparsity=0.2, mean_multiplier=5, cov_multiplier=10, filename='customizable_synthetic_dataset.txt')
