library(Matrix)
library(e1071)
library(glmnet)
set.seed(100)

# Sigmoid function
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}


# taylor expand sigmoid function
# Define the approx_sig function
approx_sig <- function(z) {
  # Constants for the Taylor series expansion at z0 = 1
  z0 <- 1
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


# Approximate
approximate <- function(z) {
  1/ (4 + z^2 + z^4/12 + z^6/360 + z^8/20160)
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
    #p_hat <- sigmoid(dot_p)
    p_hat <- approx_sig(dot_p)
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
filename = 'australian'
dataset_path <-  paste0("/Users/shaozishan/Desktop/senior_thesis/experiments/binary-class/", filename,".txt")
#dataset_path <-  paste0("/Users/shaozishan/Desktop/senior_thesis/experiments/binary-class/", filename)


# Read the data from the LIBSVM format file
lines <- readLines(dataset_path)

#cat(lines)

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

# print(labels)

# Determine the size of the feature matrix
n <- length(labels)
max_index <- max(unlist(lapply(features, function(f) as.numeric(names(f)))))
m <- max_index

# Create a sparse matrix for features
if (is.na(n) || is.na(m) || !is.numeric(n) || !is.numeric(m) || n <= 0 || m <= 0) {
  cat(m, " ")
  cat(n, " ")
  stop("n and m must be positive integers")
}

X <- Matrix::Matrix(0, nrow = n, ncol = m, sparse = TRUE)

#X <- Matrix(0, nrow = n, ncol = m, sparse = TRUE)
#X <- Matrix(0, nrow = n, ncol = m, sparse = FALSE)

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
              intercept = FALSE, maxit = 2000, thresh = 1e-16)

# Extract coefficients for the fitted model
optimal_beta <- as.matrix(coef(fit, s = lambda, exact = TRUE, drop = FALSE))
# remove the intercept
optimal_beta <- optimal_beta[-1, , drop = FALSE]



######## Testing ##########

# regular convergence
gap = 1
Beta <- rep(0, ncol(X))
r_Beta <- updateBeta(X_train, y_train, lambda, alpha, Beta, length(y_train), ncol(X), 1e-16, 1000, optimal_beta, gap)

# s-step convergence
s_gap = 1
s = 100
Beta <- rep(0, ncol(X))
s_Beta <- s_step_Beta(X_train, y_train, lambda, alpha, Beta, length(y_train), ncol(X), tol = 1e-16, maxit = 10, s, optimal_beta, s_gap)


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

#s_step_label <- paste0("s_Step_CD, s = ", s)

# Combine the beta differences into one data frame
df <- data.frame(Iteration = c(plt_Beta, plt_sBeta),
                 Difference = c(r_Beta, s_Beta),
                 Method = factor(rep(c("CD", "s_Step_CD"), 
                                     c(length(r_Beta), length(s_Beta)))))

ggplot(df, aes(x = Iteration, y = Difference, color = Method, linetype = Method)) +
  geom_line(data = df[df$Method == "CD",], aes(group = Method)) +
  geom_point(data = df[df$Method == "s_Step_CD",], size = 3) +
  #scale_y_log10() +  # Uncomment if you want the y-axis to be on a log scale
  labs(title =  paste("Convergence Plot |", filename, "s = ", s),
       x = "Iteration", y = "Squared Difference from Optimal Beta",
       color = "Method", linetype = "Method") +
  theme_minimal() + theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6)
  ) +  # Moves legend to top right
  scale_color_manual(values = c("CD" = line_color, "s_Step_CD, s = 100" = dot_color)) +
  scale_linetype_manual(values = c("CD" = "twodash", "s_Step_CD, s = 100" = "blank"))  # "blank" for points to hide line in legend

# Preparing the filename using the existing 'filename' variable
filename <- paste0('/Users/shaozishan/Desktop/senior_thesis/experiments/convergence/cd-convergence-', filename, '.pdf')

# Use ggsave() to save your plot
ggsave(filename, plot = last_plot(), dpi = 300, height = 4, width = 6, unit = 'in')

