library(Matrix)
set.seed(100)

# Sigmoid function
sigmoid <- function(z) {
  1 / (1 + exp(-z))
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
    
    # Part 1: Computing wi
    dot_p = A %*% Beta
    p_hat <- sigmoid(dot_p)
    #print(p_hat)
    W <- p_hat * (1 - p_hat)  # each row are the w_i
    # print(W) # w is correct in this case
    
    # Part 2: Compute zi
    Z = dot_p + (b - p_hat) * (1/W) # correct if class is 1/0
    print(head(Z))
    
    # Part 3: Computes g
    g = 0
    for (k in 1:m) {
      if (W[k] != 0) {
        #print((W[k] * xj[k] * (Z[k] - dot_p[k] + xj[k] * Beta[idx])))
        g <- g + (W[k] * xj[k] * (Z[k] - dot_p[k] + xj[k] * Beta[idx]))
        #print(g)
        if (is.nan(g)) {
          cat("k:", k, "\n")
          cat("w_i:", W[k], "\n")
          cat("x_ij:", xj[k], "\n")
          cat("z_i:", Z[k], "\n")
          cat("z_i_hat (dot_p):", dot_p[k], "\n")
          cat("beta_j:", Beta[idx], "\n")
          stop("Error: g is NaN.")
        }
      } else {
        print("Skipping entry with zero weight.")
      }
    }
    
    g <- g * 1/m
    
    cat("Iter:", i, "\n")
    cat("idx:", idx, "\n")
    cat("m:", m, "\n")
    cat("n:", n, "\n")
    cat("g:", g, "\n")
    #cat("cov:", cov, "\n")
    #cat("mid:", mid, "\n")
    #cat("weights:", sum(weights_sum) / m, "\n")
    
    weights_sum = xj^2 * W
    # print(weights_sum) # I think this is correct
    
    # Update Beta for the selected feature
    Beta[idx] <- softthreshold(g, lambda, alpha, sum(weights_sum))
    
  }
  
  return(Beta)
}

# Specify the path to your dataset
dataset_path <- "/Users/shaozishan/Desktop/Research/24SpringResearch/ca_reg_path/aus_small.txt"

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

Beta <- updateBeta(X, labels, lambda, alpha, Beta, n, ncol(X), 1e-10, 10)

# Make predictions
probabilities <- sigmoid(X %*% Beta)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)

# Evaluate model
accuracy <- mean(predicted_classes == labels)
cat("Accuracy:", accuracy, "\n")

# Print coefficients
print(Beta)
