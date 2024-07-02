library(Matrix)
library(e1071)
library(glmnet)
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
updateBeta <- function(A, b, lambda, alpha, Beta, m, n, tol, maxit, optimal_beta, gap) {
  
  set.seed(100)
  # so that we only capture difference of beta in certain iteration gap (adjustable)
  beta_differences <- numeric(ceiling(maxit / gap))
  counter <- 1
  
  for (i in 1:maxit) {
    
    idx <- sample(1:n, 1)  # Randomly select a feature to update
    xj <- A[, idx]
    
    # Part 1: Computing wi
    dot_p = A %*% Beta
    p_hat <- sigmoid(dot_p)
    W <- p_hat * (1 - p_hat)  # each row are the w_i
    
    # Part 2: Compute zi
    # (1) We need to figure out the entries with 0
    idx_zero <- which(W == 0)
    # (2) Computes the Z value for entries only with W greater than 0
    Z <- dot_p + (b - p_hat) * (1/W) # correct if class is 1/0 This code will have some problem if W value is 0
    Z[idx_zero] <- 0 # this is not entirely correct
    weight_xj = W * xj # this is a N x 1 vector
    
    # Part 3: Computes g
    g = 0 
    g <- 1/m * ((t(weight_xj) %*% Z) - (t(weight_xj) %*% dot_p) + (t(weight_xj) %*% xj * Beta[idx]))
    g <- as.numeric(g)
    if (is.nan(g)) {
      print("Null g value, skipped")
      next
    }
    weights_sum = xj^2 * W * 1/m
    
    # Update Beta for the selected feature
    Beta[idx] <- softthreshold(g, lambda, alpha, sum(weights_sum))
    
    #beta_differences[i] <- sqrt(sum((Beta - optimal_beta)^2))
    ## Record the beta_difference by given gap
    if (i %% gap == 0) {
      beta_differences[counter] <- sqrt(sum((Beta - optimal_beta)^2))
      counter <- counter + 1
    }
  }
  
  # Return only the computed differences
  return(beta_differences[1:(counter-1)])
}



# s-Step Coordinate Descent for logistic regression with elastic net
s_step_Beta <- function(A, b, lambda, alpha, Beta, m, n, tol, maxit, s, optimal_beta, gap) {
  
  set.seed(100)
  # so that we only capture difference of beta in certain iteration gap (adjustable)
  beta_differences <- numeric(ceiling(maxit / gap))
  counter <- 1
  
  for (i in 1:maxit) {
    
    s_idx <- sample(1:n, s, replace = TRUE)  # Randomly select a feature to update
    #cat("s_idx: ", s_idx)
    x_sj <- A[, s_idx] # N * s dimension
    
    # Part 1: Computing wi
    dot_p = A %*% Beta
    p_hat <- sigmoid(dot_p)
    W <- p_hat * (1 - p_hat)  # each row are the w_i
    
    # precompute fixed overhead
    X_bar = 1/m * t(x_sj) %*% b # this will be s * 1
    
    
    # Part 2: Computes g
    # this need to be done in a loop from 1 to s
    for (k in 1:s) {
      g = 0
      g <- X_bar[k] - 1/m * (t(x_sj[,k]) %*% p_hat) + 1/m * (t(W) %*% (x_sj[,k]^2) * Beta[s_idx[k]])
      g <- as.numeric(g)
      if (is.nan(g)) {
        print("Null g value, skipped")
        next
      }
      weights_sum = x_sj[,k]^2 * W * 1/m
      
      
      # Update Beta for the selected feature
      new_Beta <- softthreshold(g, lambda, alpha, sum(weights_sum))
      dot_p = dot_p + ((x_sj[,k]) * (new_Beta - Beta[s_idx[k]]))
      p_hat <- sigmoid(dot_p)
      W <- p_hat * (1 - p_hat)
      Beta[s_idx[k]] = new_Beta
    }
    
    ## Record the beta_difference by given gap
    if (i %% gap == 0) {
      beta_differences[counter] <- sqrt(sum((Beta - optimal_beta)^2))
      counter <- counter + 1
    }
  }
  
  # Return only the computed differences
  return(beta_differences[1:(counter-1)])
}



###### Main Program ######

# Specify the path to your dataset
filename = 'synthetic_multiclass_hard'
dataset_path <- paste0("/Users/shaozishan/Desktop/Research/24SpringResearch/ca_reg_path/", filename, ".txt")  # Update this path

# Read the data from the LIBSVM format file
lines <- readLines(dataset_path)

# Initialize vectors to store labels and a list to store features
labels <- numeric(length(lines))
features <- vector("list", length(lines))


# Parse each line to extract labels and features
# Parse each line to extract labels and features
for (i in seq_along(lines)) {
  parts <- strsplit(lines[i], " ")[[1]]
  if (length(parts) < 2) {  # Check if there are at least two parts: label and one feature
    next  # Skip malformed lines silently
  }
  label_part <- parts[1]
  numeric_label <- as.numeric(label_part)
  if (is.na(numeric_label)) {
    labels[i] <- 0  # Set NA labels to 0
  } else {
    labels[i] <- ifelse(numeric_label == 2, 0, 1)  # Convert to binary classification: class 2 as 0, others as 1
  }
  
  feature_vector <- parts[-1]
  idx_val_pairs <- lapply(feature_vector, function(feat) strsplit(feat, ":")[[1]])
  features[[i]] <- setNames(as.numeric(sapply(idx_val_pairs, `[`, 2)), 
                            as.numeric(sapply(idx_val_pairs, `[`, 1)))
}


# Determine the size of the feature matrix
n <- length(labels)
max_index <- max(unlist(lapply(features, function(f) as.numeric(names(f)))))
m <- max_index

# Create a sparse matrix for features
X <- Matrix::Matrix(0, nrow = n, ncol = m, sparse = TRUE)

# Fill the sparse matrix with feature values
for (i in 1:length(features)) {
  idx <- as.numeric(names(features[[i]]))
  vals <- as.numeric(features[[i]])
  X[i, idx] <- vals
}


# Initialize coefficients and fit the model
alpha <- 0.5
lambda <- 0.02

# Split data into training and testing sets
train_idx <- sample(1:n, 0.8 * n)  # 80% for training, adjust as needed
test_idx <- setdiff(1:n, train_idx)


# Training set
X_train <- X[train_idx, ]
y_train <- labels[train_idx]

# Testing set
X_test <- X[test_idx, ]
y_test <- labels[test_idx]



# define the optimal_beta here

fit <- glmnet(x = X_train, y = y_train, family = "binomial", nlambda = 1, alpha = alpha, lambda = c(lambda), standardize = FALSE,
              intercept = FALSE, maxit = 1000, thresh = 1e-16)

# Extract coefficients for the fitted model
optimal_beta <- as.matrix(coef(fit, s = lambda, exact = TRUE, drop = FALSE))
# remove the intercept
optimal_beta <- optimal_beta[-1, , drop = FALSE]



######## Testing ##########

# regular convergence
gap = 1
Beta <- rep(0, ncol(X))
r_Beta <- updateBeta(X_train, y_train, lambda, alpha, Beta, length(y_train), ncol(X), 1e-16, 500, optimal_beta, gap)

# s-step convergence
s_gap = 3
s = 10
Beta <- rep(0, ncol(X))
s_Beta <- s_step_Beta(X_train, y_train, lambda, alpha, Beta, length(y_train), ncol(X), tol = 1e-16, maxit = 50, s, optimal_beta, s_gap)


library(ggplot2)

# Calculate the initial difference from optimal_beta for the starting Beta (assumed to be a vector of zeros here)
initial_diff <- sqrt(sum((rep(0, ncol(X)) - optimal_beta)^2))

# Prepend the initial difference to both difference vectors
r_Beta <- c(initial_diff, r_Beta)
s_Beta <- c(initial_diff, s_Beta)

# Adjust the iteration numbers for plotting
plt_Beta <- c(0, seq(from = gap, by = gap, length.out = length(r_Beta) - 1))
plt_sBeta <- c(0, seq(from = s_gap * s, by = s_gap * s, length.out = length(s_Beta) - 1))

# Define color parameters
line_color <- "blue2"
dot_color <- "red"

s_step_label <- paste0("s_Step_CD, s = ", s)

# Combine the beta differences into one data frame
df <- data.frame(Iteration = c(plt_Beta, plt_sBeta),
                 Difference = c(r_Beta, s_Beta),
                 Method = factor(rep(c("CD", "s_Step_CD"), 
                                     c(length(r_Beta), length(s_Beta)))))

ggplot(df, aes(x = Iteration, y = Difference, color = Method, linetype = Method)) +
  geom_line(data = df[df$Method == "CD",], aes(group = Method)) +
  geom_point(data = df[df$Method == "s_Step_CD",], size = 3) +
  #scale_y_log10() +  # Uncomment if you want the y-axis to be on a log scale
  labs(title = paste("Convergence Plot |", filename, "s = ", s),
       x = "Iteration", y = "Squared Difference from Optimal Beta",
       color = "Method", linetype = "Method") +
  theme_minimal() + theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6)
  ) +  # Moves legend to top right
  scale_color_manual(values = c("CD" = line_color, "s_Step_CD" = dot_color)) +
  scale_linetype_manual(values = c("CD" = "twodash", "s_Step_CD" = "blank"))  # "blank" for points to hide line in legend

 
#ggplot(df, aes(x = Iteration, y = Difference, color = Method)) +
#  geom_line(data = df[df$Method == "CD",], aes(group = Method), ,linetype="twodash") +  # Only plot lines for r-Beta
#  geom_point(data = df[df$Method == "s_step_CD",], size = 1.5) +  # Only plot points for s-Beta
#  #scale_y_log10() +  # Uncomment if you want the y-axis to be on a log scale
#  labs(title = "Convergence of Beta Coefficients Across Methods",
#       x = "Iteration", y = "Squared Difference from Optimal Beta") +
#  theme_minimal() +
#  theme(legend.title = element_blank())



# Preparing the filename using the existing 'filename' variable
filename <- paste0('/Users/shaozishan/Desktop/senior_thesis/experiments/convergence/multi-cd-convergence-', filename, '.pdf')

# Use ggsave() to save your plot
ggsave(filename, plot = last_plot(), dpi = 300, height = 4, width = 6, unit = 'in')

