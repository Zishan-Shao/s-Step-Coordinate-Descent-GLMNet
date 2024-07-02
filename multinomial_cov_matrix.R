library(Matrix)
set.seed(100)

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
    cov_xjy = sum(xj * y)
    cov_jk = (t(xj) %*% A) %*% Beta
    
    # Part 2: Computes g
    g = 0
    # g <- 1/m * (t(weight_xj) %*% (Z - dot_p) + t(weight_xj) %*% xj * Beta[idx]) # naive updates are correct
    g <- 1/m * (cov_xjy - cov_jk + (t(xj) %*% xj * Beta[idx]))
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
    
    weights_sum = xj^2 * 1/m
    # print(weights_sum) # I think this is correct
    
    # Update Beta for the selected feature
    Beta[idx] <- softthreshold(g, lambda, alpha, sum(weights_sum))

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


Beta <- updateBeta(X_train, y_train, lambda, alpha, Beta, length(y_train), ncol(X), 1e-10, 3000)

# Make predictions
probabilities <- sigmoid(X %*% Beta)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)

# Evaluate model
accuracy <- mean(predicted_classes == labels)
cat("Accuracy:", accuracy, "\n")

# Print coefficients
print(Beta)
