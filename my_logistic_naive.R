library(Matrix)
set.seed(100)

# Sigmoid function
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

# Approximate cross entropy

approximate_cross_entropy <- function(z) {
  #1/ (4 + z^2 + z^4/12 + z^6/360 + z^8/20160)
  1/ (4 + z^2)
}



# approximate the sigmoid function
approximate_sigmoid <- function(z) {
  #1/ (2 - z + z^2/2 - z^3/6 + z^4/24 - z^5/120 + z^6/720)
  1/ (2 - z + z^2/2 - z^3/6 + z^4/24 - z^5/120 + z^6/720)
  #1/ (2 - z + z^2/2)
}

approx_sig <- function(z) {
  # Constants for the Taylor series expansion at z0 = 1
  z0 <- 0.01
  e <- exp(1)
  S0 <- e / (e + 1)
  S1 <- 1 / (e + 1)^2
  S2 <- (-e + 2) / (2 * (e + 1)^3)
  S3 <- (-6 * e^2 + 6 * e + 1) / (6 * (e + 1)^4)
  S4 <- (-36 * e^3 - e + 24 + 14 * e^2) / (24 * (e + 1)^5)
  
  # Taylor series expansion
  result <- S0 + S1 * (z - z0) + S2 * (z - z0)^2 + S3 * (z - z0)^3 + S4 * (z - z0)^4
  return(result)
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
    z_hat <- sigmoid(dot_p)
    #cat("sigmoid")
    #print(head(p_hat))
    p_hat <- approx_sig(dot_p) # try to approximate 1/ (1- e^-z)
    cat("diff")
    #print(head(p_hat))
    print(sum((p_hat - z_hat)^2))
    #exit(0)
    #print(p_hat)
    W <- p_hat * (1 - p_hat)  # each row are the w_i
    #W <- approximate_cross_entropy(dot_p)
    # print(W) # w is correct in this case
    
    # Part 2: Compute zi
    Z = dot_p + (b - p_hat) * (1/W) # correct if class is 1/0
    #print(head(Z))

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
    
    weights_sum = 1/m * xj^2 * W
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


Beta <- updateBeta(X_train, y_train, lambda, alpha, Beta, length(y_train), ncol(X), 1e-10, 200)

# Print coefficients
print(Beta)


# use glmnet to find out optimal solution
fit <- glmnet(x = X_train, y = y_train, family = "binomial", nlambda = 1, alpha = alpha, lambda = c(lambda), standardize = FALSE,
              intercept = FALSE, maxit = 3000, thresh = 1e-16)

# Extract coefficients for the fitted model
optimal_beta <- as.matrix(coef(fit, s = lambda, exact = TRUE, drop = FALSE))
# remove the intercept
optimal_beta <- optimal_beta[-1, , drop = FALSE]

print(optimal_beta)

cat("relative error:", sum((optimal_beta - Beta)^2))

