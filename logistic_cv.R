library(e1071)
library(glmnet)
library(Matrix)

# Read the dataset
#data <- read.matrix.csr(dataset_path) # not working

# Adjust the path to where your dataset is located
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
  # Split each feature on ":" to separate index and value, then store in a named vector
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


# At this point, X is your feature matrix in sparse format, and 'labels' contains your labels

# Proceed with any analysis or modeling, e.g., fitting a model with glmnet
# Install and load glmnet if you haven't already
# install.packages("glmnet")
library(glmnet)

# Assume we are doing logistic regression (for binary classification problems)
# Define alpha and lambda for elastic net; values should be chosen based on your specific needs
alpha <- 0.5  # Mixing parameter between LASSO (1) and ridge (0)
lambda <- 0.02  # Regularization strength

# Fit the model
fit <- glmnet(x = X, y = labels, family = "binomial", alpha = alpha)

# Optionally, use cross-validation to find an optimal lambda value
cv_fit <- cv.glmnet(X, labels, family = "binomial", type.measure = "class", alpha = alpha)

# Print optimal lambda value (from cross-validation)
print(paste("Optimal lambda:", cv_fit$lambda.min))

# Extract the optimal lambda value chosen by cross-validation
best_lambda <- cv_fit$lambda.min

# Make predictions using cv_fit and best_lambda
predictions <- predict(cv_fit, newx = X, type = "response", s = "lambda.min")

# Convert predictions to binary classes
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Ensure labels are correctly represented as numeric (0 and 1) if not already
actual_classes <- as.numeric(labels)  # Adjust based on your labels representation



##### Print out the results #####

# Calculate and print accuracy
accuracy <- mean(predicted_classes == actual_classes)
cat("Accuracy:", accuracy, "\n")

# Print the best lambda value
cat("Best lambda:", best_lambda, "\n")

# Coefficients at the best lambda
betas <- coef(cv_fit, s = "lambda.min")
cat("Coefficients (Betas):\n")
print(betas)


