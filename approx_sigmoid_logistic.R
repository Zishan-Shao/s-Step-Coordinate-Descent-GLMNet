library(Matrix)
library(numDeriv)

set.seed(100)

# Sigmoid function
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

# Approximate sigmoid function using Taylor series expansion up to the 5th term
approx_sig <- function(z, mean_z) {
  # Constants for the Taylor series expansion around mean_z
  S0 <- sigmoid(mean_z)
  S1 <- S0 * (1 - S0)
  S2 <- S1 * (1 - 2 * S0) / 2
  S3 <- S2 * (1 - 3 * S0 + 3 * S0^2) / 3
  S4 <- S3 * (1 - 4 * S0 + 6 * S0^2 - 4 * S0^3) / 4
  
  # Taylor series expansion
  result <- S0 + S1 * (z - mean_z) + S2 * (z - mean_z)^2 + S3 * (z - mean_z)^3 + S4 * (z - mean_z)^4
  return(result)
}

approx_exp <- function(x) {
  
  # Return the approximated value at x
  return(2)
}


# Soft thresholding for elastic net penalty
softthreshold <- function(g, lambda, alpha, w) {
  rho <- lambda * alpha
  if (g > rho) return ((g - rho) / (w + lambda * (1 - alpha)))
  if (g < -rho) return ((g + rho) / (w + lambda * (1 - alpha)))
  return (0)
}

# Coordinate descent for logistic regression with elastic net
updateBeta <- function(A, b, lambda, alpha, Beta, m, n, tol, maxit) {
  for (i in 1:maxit) {
    idx <- sample(1:n, 1)  # Randomly select a feature to update
    xj <- A[, idx]
    
    # Compute dot product of A and Beta
    dot_p <- A %*% Beta
    z_hat <- sigmoid(dot_p)
    
    # Approximate sigmoid function
    #p_hat <- approx_sig(dot_p)
    #mean_z <- mean(dot_p)  # Calculate mean of the linear combination
    #p_hat <- approx_sig(dot_p, mean_z)  # Approximate sigmoid using Taylor series around mean
    p_hat <- approx_exp_euler(dot_p)  # Approximate sigmoid using Taylor series around mean
    
    cat("Iter:", i, "Diff:", sum((p_hat - z_hat)^2), "\n")
    
    # Compute weights
    W <- p_hat * (1 - p_hat)
    
    # Compute Z
    Z <- dot_p + (b - p_hat) / W
    
    # Compute gradient g
    g <- 0
    for (k in 1:m) {
      if (W[k] != 0) {
        g <- g + (W[k] * xj[k] * (Z[k] - dot_p[k] + xj[k] * Beta[idx]))
      }
    }
    g <- g * (1 / m)
    
    # Compute weights_sum
    weights_sum <- 1 / m * xj^2 * W
    
    # Update Beta for the selected feature
    Beta[idx] <- softthreshold(g, lambda, alpha, sum(weights_sum))
  }
  
  return(Beta)
}

# Test the coordinate descent algorithm
# Specify the path to your dataset
dataset_path <- "/Users/shaozishan/Desktop/Research/24SpringResearch/ca_reg_path/australian.txt"

# Read the data from the LIBSVM format file
lines <- readLines(dataset_path)

# Initialize vectors to store labels and a list to store features
labels <- numeric(length(lines))
features <- vector("list", length(lines))

# Parse each line to extract labels and features
for (i in seq_along(lines)) {
  parts <- strsplit(lines[i], " ")[[1]]
  labels[i] <- as.numeric(parts[1])  # Extract label
  if(labels[i] == -1) {
    labels[i] = 0;
  }
  feature_vector <- parts[-1]  # Extract features
  idx_val_pairs <- lapply(feature_vector, function(feat) strsplit(feat, ":")[[1]])
  features[[i]] <- setNames(as.numeric(sapply(idx_val_pairs, `[`, 2)), 
                            as.numeric(sapply(idx_val_pairs, `[`, 1)))
}

print(labels)

# Determine the size of the feature matrix
n <- length(labels)
max_index <- max(unlist(lapply(features, function(f) as.numeric(names(f)))))
m <- max_index

# Create a sparse matrix for features
X <- Matrix(0, nrow = n, ncol = m, sparse = TRUE)

# Fill the sparse matrix with feature values
for (i in 1:length(features)) {
  idx <- as.numeric(names(features[[i]]))
  vals <- as.numeric(features[[i]])
  X[i, idx] <- vals
}


# Initialize coefficients and fit the model
alpha <- 0.5
lambda <- 0.02
Beta <- rep(0, ncol(X))

# Split data into training and testing sets
train_idx <- sample(1:n, 0.8 * n)  # 80% for training, adjust as needed
test_idx <- setdiff(1:n, train_idx)


# Training set
X_train <- X[train_idx, ]
y_train <- labels[train_idx]

# Testing set
X_test <- X[test_idx, ]
y_test <- labels[test_idx]


Beta <- updateBeta(X_train, y_train, lambda, alpha, Beta, length(y_train), ncol(X), 1e-10, 500)
print(Beta)

# Error calculation
error <- sum((approx_values - sigmoid_values)^2)
print(paste("Total error:", error))
