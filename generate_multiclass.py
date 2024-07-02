import numpy as np

def generate_libsvm_dataset(K, m, n, sparsity=0.1, filename='dataset.txt'):
    """
    Generates a dataset in LIBSVM format with K classes, m observations, and n features,
    with specified sparsity and balanced class distribution.
    
    Args:
    K (int): Number of classes.
    m (int): Total number of observations.
    n (int): Number of features.
    sparsity (float): Fraction of features to set as zero (sparsity).
    filename (str): Path and name of the file to save the dataset.
    
    Returns:
    None: Saves the dataset to a text file in LIBSVM format.
    """
    # Calculate base count and adjust for 20% variability
    base_count = m // K
    min_count = int(base_count * 0.8)
    max_count = int(base_count * 1.2)
    
    # Adjust counts to exactly sum up to m
    counts = np.random.randint(min_count, max_count+1, size=K)
    while sum(counts) != m:
        counts = np.random.randint(min_count, max_count+1, size=K)
        if sum(counts) > m:
            counts[np.argmax(counts)] -= sum(counts) - m
        elif sum(counts) < m:
            counts[np.argmin(counts)] += m - sum(counts)
    
    # Open the file to write
    with open(filename, 'w') as file:
        for i in range(K):
            # Generate data for each class
            class_features = np.random.randn(counts[i], n)
            # Introduce sparsity
            mask = np.random.rand(counts[i], n) < sparsity
            class_features[mask] = 0
            
            # Write each observation to the file
            for features in class_features:
                # Format the line as label followed by non-zero features: index:value
                line = str(i)
                for index, value in enumerate(features, start=1):
                    if value != 0:
                        line += f" {index}:{value:.4f}"
                line += "\n"
                file.write(line)

    print(f'Dataset with {m} observations, {n} features, and {K} classes saved to {filename}')

# Example usage
generate_libsvm_dataset(K=3, m=2000, n=16, sparsity=0.2, filename='/Users/shaozishan/Desktop/Research/24SpringResearch/ca_reg_path/synthetic_multiclass_dataset.txt')
