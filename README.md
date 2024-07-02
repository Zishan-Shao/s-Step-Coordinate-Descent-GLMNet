# s-Step-Coordinate-Descent-GLMNet
Derive and implement a s-Step variant of coordinate descent in computing the GLMNet problem

This project was based on the paper: Friedman J, Hastie T, Tibshirani R. Regularization Paths for Generalized Linear Models via Coordinate Descent. J Stat Softw. 2010;33(1):1-22. PMID: 20808728; PMCID: PMC2929880.

--------------------------------------------------------------------------------------------------

Algorithm Implementation:

my_logistic_covariance.R : coordinate descent algorithm with covariance updates in solving logistic regression problem with elastic net regularization, mentioned in the referred paper

my_logistic_covariance.R : same problem but with matrix implemented

my_logistic_naive.R : coordinate descent on regularized logistic regression but with naive updates mentioned in referred paper

my_logistic_matrix_naive.R : coordinate descent on regularized logistic regression with naive updates, implemented in matrix operations, mentioned in referred paper

multinomial_cov_matrix.R : coordinate descent on regularized multinomial regression but with naive updates mentioned in referred paper

s_step_logistic_cov_matrix.R : s-Step variant of coordinate descent algorithm


--------------------------------------------------------------------------------------------------


Training Data Generate

generate_multiclass.R : Generate Multi-class datasets with randomness

generate_hard_tunable : Adding specific features of datasets, such as sparsity

