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
  beta_differences <- numeric(maxit)  # To store differences for each iteration
  optimal_beta <- c(-0.0509560268, -0.0269407763, -0.0238840738, -0.1037091940, 0.0991171915, -0.0059652048,
                    0.1481015125, 1.8942074228, 0.0000000000, 0.1994707131, 0.0000000000, -0.6463024595,
                    -0.0023842876, 0.0004098763)
  
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
    # (1) We need to figure out the entries with 0
    idx_zero <- which(W == 0)
    # (2) Computes the Z value for entries only with W greater than 0
    Z <- dot_p + (b - p_hat) * (1/W) # correct if class is 1/0 This code will have some problem if W value is 0
    Z[idx_zero] <- 0 # this is not entirely correct
    print(head(Z))
    weight_xj = W * xj # this is a N x 1 vector
    
    # Part 3: Computes g
    g = 0
    # g <- 1/m * (t(weight_xj) %*% (Z - dot_p) + t(weight_xj) %*% xj * Beta[idx]) # naive updates are correct
    g <- 1/m * ((t(weight_xj) %*% Z) - (t(weight_xj) %*% dot_p) + (t(weight_xj) %*% xj * Beta[idx]))
    cat("problem: ", ((t(weight_xj) %*% Z) - (t(weight_xj) %*% dot_p))[1], "\n")
    g <- as.numeric(g)
    if (is.nan(g)) {
      print("Null g value, skipped")
      next
    }
    
    
    cat("Iter:", i, "\n")
    cat("idx:", idx, "\n")
    cat("g:", g, "\n")
    #cat("cov:", cov, "\n")
    #cat("mid:", mid, "\n")
    #cat("weights:", sum(weights_sum) / m, "\n")
    
    weights_sum = xj^2 * W * 1/m
    # print(weights_sum) # I think this is correct
    
    # Update Beta for the selected feature
    Beta[idx] <- softthreshold(g, lambda, alpha, sum(weights_sum))
    
    beta_differences[i] <- sqrt(sum((Beta - optimal_beta)^2))
    

  }
  
  return(Beta)
}

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


Beta <- updateBeta(X_train, y_train, lambda, alpha, Beta, length(y_train), ncol(X), 1e-10, 1000)

# Make predictions
probabilities <- sigmoid(X %*% Beta)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)

# Evaluate model
accuracy <- mean(predicted_classes == labels)
cat("Accuracy:", accuracy, "\n")

# Print coefficients
print(Beta)
