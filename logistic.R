# Load necessary libraries
library(e1071)
library(glmnet)
library(Matrix)
set.seed(100)

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
  feature_vector <- parts[-1]  # Extract features
  idx_val_pairs <- lapply(feature_vector, function(feat) strsplit(feat, ":")[[1]])
  features[[i]] <- setNames(as.numeric(sapply(idx_val_pairs, `[`, 2)), 
                            as.numeric(sapply(idx_val_pairs, `[`, 1)))
}

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

# Fit the model with elastic net regularization
alpha <- 0.5  # Elastic Net mixing parameter (0 = Ridge, 1 = Lasso)
lambda <- 0.02  # Regularization strength

# Define training and testing sets
train_idx <- sample(1:n, 0.8 * n)  # 80% for training, adjust as needed
test_idx <- setdiff(1:n, train_idx)

# print(train_idx) # the train-test sets are the same

# Training set
X_train <- X[train_idx, ]
y_train <- labels[train_idx]

# Testing set
X_test <- X[test_idx, ]
y_test <- labels[test_idx]

fit <- glmnet(x = X_train, y = y_train, family = "binomial", nlambda = 1, alpha = alpha, lambda = c(lambda), standardize = FALSE,
              intercept = FALSE, maxit = 2000, thresh = 1e-16)


# Predict probabilities
probabilities <- predict(fit, newx = X_test, type = "response", s = lambda)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)

# Calculate accuracy
actual_classes <- as.numeric(y_test)  # Adjust this as necessary
accuracy <- mean(predicted_classes == actual_classes)
cat("Accuracy:", accuracy, "\n")

# Print the best lambda and coefficients
cat("Lambda used:", lambda, "\n")
betas <- coef(fit, s = lambda)
cat("Coefficients (Betas):\n")
print(betas)
