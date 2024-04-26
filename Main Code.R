# R SCRIPT FOR ICA OF STAT0044
# STUDENT NUMBER: 21000790

################### GUIDANCE FOR RUNNING THIS SCRIPT #########################################

# STRUCTURE:
# This script is structured according to the ICA instruction and includes:
# - Question 1: Setup input data
# - Question 2: Simulate cauchit data
# - Question 3: Perform Rejection Sampling
# - Question 4: Execute MCMC on 4 different posteriors (Posterior 1, 2, 3, 4)

# SAMPLING FUNCTIONS:
# Before sampling, three different SAMPLING FUNCTIONS are defined:
# - RWM_iid_componentwise_adaptive
# - RWM_multi
# - precondition_RWM_multi
# These functions are used in each posterior section with automated step size adjustments (no manual tuning required).

# MANUAL TASKS:
# The only manual tasks required to reproduce the results are:
# Step 1: Modify the candidate distribution in the SAMPLING FUNCTIONS as specified in Table 1 and Table 4 of the report.
#         (Note: For preconditioning, update the candidate distribution in the 'RWM_multi' function, as it is a helper for 'precondition_RWM_multi')
# Step 2: Execute the sampling for the desired model.
# Step 3: Run the corresponding diagnostic code (trace plot, ESS, posterior mean and CI) for the chosen sampling method, results, and dataset.

# EXAMPLE:
# To reproduce the result for a candidate: Multivariate Student-t with degree of freedom 1 and with preconditioning in Posterior 1 (Logistic + IID-N Prior):
# Step 0: Run all Question 1, Question 2 and Question 3 to obtain all datasets
# Step 1: Go to 'SAMPLING FUNCTION', In 'RWM_multi', change the candidate distribution to 'as.vector(rmvt(1, sigma = V, df=1))' (look for the "CANDIDATE DISTRIBUTION, UPDATE AS GUIDED ABOVE" comment). 
# Then run the 'RWM_multi' function and 'precondition_RWM_multi' function. 
# Step 2: In section 'Posterior 1: Logistic + NORMAL IID Prior', run the seed and 'Multivariate + Preconditioning' code chunk to get 'results_model_1_multi_pcond'.
# Step 3: In the 'DIAGNOSIS CODE for precondition_RWM_multi', assign 'results_model_1_multi_pcond' to 'results_pcond' and 'Y_logit' to 'y_true'.

###########################################################################################

# Install and load necessary packages
if (!require("coda")) install.packages("coda", dependencies = TRUE)
library(coda)
library(mvtnorm)

###################### QUESTION 1 ####################

set.seed(0790)

n <- 150 # number of observations
d <- 10 # number of beta parameters

# create matrix to populate with covariates
X <- matrix(nrow = n, ncol = d)
X[,1] <- rep(1, n) # first column is an intercept

# create base uncorrelated random numbers to turn into x_i's
z <- matrix(rnorm(n*(d-1)), nrow = n, ncol = d-1)

# create x_i's (ith column of matrix corresponds to variable x_i)
X[,2] <- z[,1]
X[,3] <- z[,1] + 0.2*z[,2]
X[,4] <- 0.5*z[,3]
X[,5] <- z[,4]
X[,6] <- 2*z[,5] + 20*z[,6]
X[,7] <- z[,6]
X[,8] <- 0.5 * (z[,7] + z[,4] + z[,8] + z[,1])
X[,9] <- z[,8] + 10*z[,4]
X[,10] <- z[,5] + 0.5*z[,9]

# create true beta values
beta <- seq(-2,2, length = 10)

###################### QUESTION 2 ####################

# draw n epsilon_i's from the Cauchy distribution (t distribution with df = 1)
epsilon_cauchit <- rt(n, df = 1)

Y_star_cauchit <- X %*% beta + epsilon_cauchit

Y_cauchit <- ifelse(Y_star_cauchit > 0, 1, 0)


###################### QUESTION 3 ####################

f_logistic <- function(x) {
  return(exp(-x) / (1 + exp(-x))^2)
}

f_cauchy <- function(x) {
  return(1 / (pi * (1 + x^2)))
}

rejection_sampling <- function(sample_size, M) {
  samples <- numeric(sample_size)
  count <- 0
  total_iterations <- 0
  while(count < sample_size) {
    x_prop <- rt(1, df = 1)
    u <- runif(1)
    total_iterations <- total_iterations + 1

    log_ratio <- log(f_logistic(x_prop)) - log(f_cauchy(x_prop))
    if (!is.na(log_ratio) && log(M) + log(u) <= log_ratio) {
      samples[count + 1] <- x_prop
      count <- count + 1
    }
  }
  list(samples = samples, total_iterations = total_iterations)
}

# finding optimal ratio of M
x_vals <- seq(-10, 10, length.out = 10000)
ratio_M <- f_logistic(x_vals) / f_cauchy(x_vals)
opt_M <- max(ratio_M)
opt_x <- x_vals[which.max(ratio_M)]

# Figure 1a: Ratio of Logistic to Cauchy Density
png("ratio_plot.png", width = 600, height = 400)
par(mar = c(5.1, 5.1, 4.1, 2.1))  # Adjust margins (bottom, left, top, right)
plot(x_vals, ratio_M, type = 'l', col = 'blue',
     main = " ",
     xlab = "Input values", ylab = "Ratio M (Logistic / Cauchit)",
     lwd = 2,cex.axis = 1.8, cex.lab = 1.8, cex.main = 1.8,  ylim = c(0, 1.9))

points(opt_x, opt_M, col = "red", pch = 19, cex = 1.8)
text(opt_x, opt_M, labels = sprintf("Max M: %.2f", opt_M), pos = 2, cex = 1.8, col = "red")
dev.off()

# running rejection sampling using optimal M
set.seed(0790) 
result <- rejection_sampling(150, opt_M)
epsilon_samples <- result$samples
efficiency <- 150 / result$total_iterations
print(paste("Efficiency of using rejection sampling to sample logistic is: ", efficiency))

# Figure 1b: Comparison of Rejection Sampling Density and True Logistic Density
png("sampling_density_plot.png", width = 600, height = 400)
par(mar = c(5.1, 5.1, 4.1, 2.1))  # Adjust margins (bottom, left, top, right)
hist(epsilon_samples, breaks = 50, freq = FALSE, 
     main = "", 
     xlab = "Epsilon", ylab = "Density", cex.main = 1.8, cex.lab = 1.8, cex.axis = 1.8, col = "lightblue")
curve(f_logistic, from = min(epsilon_samples), to = max(epsilon_samples), col = "red", lwd = 2, add = TRUE)
legend("topright", legend = c("True Value", "Rejection Sampling"), 
       col = c("red", "lightblue"), lwd = 4, cex = 1.4)
dev.off()

# generating Logit Observations using rejection sampled epsilons, used for next Question.

Y_star_logit <- X %*% beta + epsilon_samples

Y_logit <- ifelse(Y_star_logit > 0, 1, 0)

###################### QUESTION 4 ####################
####### SAMPLING FUNCTION ####### 

RWM_iid_componentwise_adaptive <- function(logpi, nits, x_curr, burn_in = 0.5) {
  logpi_curr <- logpi(x_curr)
  d <- length(x_curr) 
  accepted <- rep(0, d) 
  h <- rep(1, d)
  x_store <- matrix(nrow = nits, ncol = d)
  
  n_burn_in <- floor(nits * burn_in) # number of iterations in burn in phase
  
  for (i in 1:nits) {
    for (j in 1:d) {
      # propose a candidate move for component j
      x_prop <- x_curr
      x_prop[j] <- x_curr[j] + h[j] * rt(1, df=0.1) # CANDIDATE DISTRIBUTION, UPDATE AS GUIDED ABOVE
      
      logpi_prop <- logpi(x_prop)
      # check if logpi_prop is finite before acceptance
      if (!is.finite(logpi_prop)) {
        next # skip this iteration if logpi_prop is not finite, for numerical stability management
      }
      
      # accept-reject step for component j
      loga <- logpi_prop - logpi_curr
      if (log(runif(1)) < loga) {
        x_curr[j] <- x_prop[j] 
        logpi_curr <- logpi_prop
        accepted[j] <- accepted[j] + 1
      }
    }
    x_store[i,] <- x_curr
    
    # adapt step sizes every 100 iterations so that it stays between 0.36 and 0.44
    if (i <= n_burn_in) {
      if (i %% 100 == 0) { 
        for (j in 1:d) {
          ratio <- accepted[j] / i
          if (ratio > 0.44) {
            h[j] <- h[j] * 1.1
          } else if (ratio < 0.36) {
            h[j] <- h[j] * 0.9
          }
        }
      }
    }
  }
  
  return(list(x_store = x_store, acceptance_rate = accepted / nits, final_step_sizes = h))
}

RWM_multi <- function(logpi, nits, h, x_curr, V = diag(rep(1, length(x_curr))), adapt_interval = 50) {
  logpi_curr <- logpi(x_curr)
  d <- length(x_curr)
  accepted <- 0  # counter used for tuning step-size
  total_accepted <- 0  # counter used for overall acceptance rate
  x_store <- matrix(nrow = nits, ncol = d)
  
  for (i in 1:nits) {
    x_prop <- x_curr + h * as.vector(rmvt(1, sigma = V, df=1))  # CANDIDATE DISTRIBUTION, UPDATE AS GUIDED ABOVE
    logpi_prop <- logpi(x_prop)
    
    if (log(runif(1)) < logpi_prop - logpi_curr) {
      x_curr <- x_prop
      logpi_curr <- logpi_prop
      accepted <- accepted + 1
      total_accepted <- total_accepted + 1
    }
    x_store[i, ] <- x_curr
    
    # adapt step sizes every 50 iteration (tested to be the best) so that it stays near to 0.234
    if (i %% adapt_interval == 0) {
      accept_rate <- accepted / adapt_interval
      if (accept_rate < 0.234) {
        h <- h * 0.9  
      } else if (accept_rate > 0.234) {
        h <- h * 1.1 
      }
      accepted <- 0  
    }
  }
  return(list(x_store = x_store, acceptance_rate = total_accepted / nits, final_h = h))
}

precondition_RWM_multi <- function(logpi, initial_x, initial_h, nits, num_precondition, adapt_interval = 50) {
  
  d <- length(initial_x)
  V <- diag(rep(1, d))
  h <- initial_h
  x_curr <- initial_x
  
  # storing results of each preconditioning step
  results_list <- vector("list", num_precondition)
  
  for (i in 1:num_precondition) {
    
    mcmc_result <- RWM_multi(logpi = logpi, nits = nits, h = h, x_curr = x_curr, V = V, adapt_interval = adapt_interval)
    
    print(paste("Preconditioning step", i, ": Final acceptance rate =", mcmc_result$acceptance_rate * 100, "%, h =", mcmc_result$final_h))
    
    # update parameters for the next iteration
    x_curr <- mcmc_result$x_store[nits, ]  # use the last sample as the new starting point
    V <- cov(mcmc_result$x_store)          # estimate the covariance matrix
    h <- mcmc_result$final_h               # use latest step size
    
    # Store results
    results_list[[i]] <- mcmc_result
  }
  return(results_list)
}


####### Posterior 1: Logistic + NORMAL IID Prior ####### 

log_posterior_logit_iid <- function(beta) {
  log_likelihood <- sum(dbinom(Y_logit, size = 1, prob = plogis(X %*% beta), log = TRUE))
  log_prior <- sum(dnorm(beta, 0, 1, log = TRUE))
  return(log_likelihood + log_prior)
}

# IID and Component Wise 
set.seed(0790) 
results_model_1_componentwise<- RWM_iid_componentwise_adaptive(
  logpi = log_posterior_logit_iid,
  nits = 20000,
  x_curr = rep(0, d) 
)

# Multivariate
set.seed(0790)
results_model_1_multi<- RWM_multi(logpi = log_posterior_logit_iid, nits = 20000, h = 0.1, x_curr = rep(0, d))

# Multivariate + Preconditioning
set.seed(0790)
results_model_1_multi_pcond <- precondition_RWM_multi(logpi = log_posterior_logit_iid, initial_x = rep(0, 10), 
                                                      initial_h = 0.5, nits = 10000, num_precondition = 5)


####### Posterior 2: Logistic + UIP Prior ####### 

log_posterior_logit_UIP <- function(beta) {
  log_likelihood <- sum(dbinom(Y_logit, size = 1, prob = plogis(X %*% beta), log = TRUE))
  fisher_info <- n * solve(t(X) %*% X)
  log_prior <- log(dmvnorm(beta, sigma = fisher_info))
  
  return(log_likelihood + log_prior)
}


# IID and Component Wise 
set.seed(0790) 
results_model_2_componentwise<- RWM_iid_componentwise_adaptive(
  logpi = log_posterior_logit_UIP,
  nits = 20000,
  x_curr = rep(0, d) 
)

# Multivariate
set.seed(0790)
results_model_2_multi<- RWM_multi(
  logpi = log_posterior_logit_UIP,
  nits = 20000,
  h = 0.044,
  x_curr = rep(0, d) 
)

# Multivariate + Preconditioning
set.seed(0790)
results_model_2_multi_pcond <- precondition_RWM_multi(logpi = log_posterior_logit_UIP, initial_x = rep(0, 10), 
                                                      initial_h = 1, nits = 10000, num_precondition = 5)


####### Posterior 3: Cauchit + NORMAL IID Prior ####### 

log_posterior_cauchit_iid <- function(beta) {
  log_likelihood <- sum(dbinom(Y_cauchit, size = 1, prob = pt(X %*% beta, df=1), log = TRUE))
  log_prior <- sum(dnorm(beta, 0, 1, log = TRUE))
  return(log_likelihood + log_prior)
}


# IID and Component Wise 
set.seed(0790)
results_model_3_componentwise <- RWM_iid_componentwise_adaptive(
  logpi = log_posterior_cauchit_iid,
  nits = 20000,
  x_curr = rep(0, d) 
)

# Multivariate
set.seed(0790)
results_model_3_multi<- RWM_multi(
  logpi = log_posterior_cauchit_iid,
  nits = 20000,
  h = 0.1,
  x_curr = rep(0, d) 
)

# Multivariate + Preconditioning
set.seed(0790)
results_model_3_multi_pcond <- precondition_RWM_multi(logpi = log_posterior_cauchit_iid, initial_x = rep(0, 10), initial_h = 0.5, nits = 10000, num_precondition = 5)



####### Posterior 4: Cauchit + UIP Prior ####### 

log_posterior_cauchit_UIP <- function(beta) {
  log_likelihood <- sum(dbinom(Y_cauchit, size = 1, prob = pt(X %*% beta, df=1), log = TRUE))
  fisher_info <- n * solve(t(X) %*% X)
  log_prior <- log(dmvnorm(beta, sigma = fisher_info))
  
  return(log_likelihood + log_prior)
}


# IID and Component Wise 
set.seed(0790)
results_model_4_componentwise <- RWM_iid_componentwise_adaptive(
  logpi = log_posterior_cauchit_UIP,
  nits = 20000,
  x_curr = rep(0, d) 
)

# Multivariate
set.seed(0790)
results_model_4_multi <- RWM_multi(
  logpi = log_posterior_cauchit_UIP,
  nits = 20000,
  h = 0.044,
  x_curr = rep(0, d) 
)

# Multivariate + Preconditioning
set.seed(0790)
results_model_4_multi_pcond <- precondition_RWM_multi(logpi = log_posterior_cauchit_UIP, initial_x = rep(0, 10), initial_h = 0.8, nits = 10000, num_precondition = 5)



###################### DIAGNOSIS CODE for RWM_iid_componentwise_adaptive ####################

# available options for result_to_diagnose:
# - results_model_1_componentwise
# - results_model_2_componentwise
# - results_model_3_componentwise
# - results_model_4_componentwise

# available options for y_true:
# - Y_logit
# - Y_cauchit

result_to_diagnose_componentwise <- results_model_1_componentwise
y_true <- Y_logit

# TRACE PLOTS
for (i in 1:ncol(result_to_diagnose_componentwise$x_store)) {
  plot(result_to_diagnose_componentwise$x_store[,i], type = 'l', col = i, main = paste("Beta", i), xlab = "Iteration", ylab = "Value")
}


# calculate Brier score
set.seed(0790)
p_forecast_diagnose_componentwise <- plogis(X %*% colMeans(tail(result_to_diagnose_componentwise$x_store, 10000)))
brier_score_diagnose_componentwise <- mean((y_true - p_forecast_diagnose_componentwise)^2)
print(paste("Brier score: ",brier_score_diagnose_componentwise))

# posterior means and ESS calculations are not presented here as they are not used in the report for this class of candidates.

###################### DIAGNOSIS CODE for RWM_multi ####################

# available options for result_to_diagnose:
# - results_model_1_multi
# - results_model_2_multi
# - results_model_3_multi
# - results_model_4_multi

# available options for y_true:
# - Y_logit
# - Y_cauchit

result_to_diagnose_multi <- results_model_1_multi
y_true <- Y_logit

# TRACE PLOTS
for (i in 1:ncol(result_to_diagnose_multi$x_store)) {
  plot(result_to_diagnose_multi$x_store[,i], type = 'l', col = i, main = paste("Beta", i), xlab = "Iteration", ylab = "Value")
}


# calculate Brier score
set.seed(0790)
p_forecast_diagnose_multi <- plogis(X %*% colMeans(tail(result_to_diagnose_multi$x_store, 10000)))
brier_score_diagnose_multi <- mean((y_true - p_forecast_diagnose_multi)^2)
print(paste("Brier score: ",brier_score_diagnose_multi))



###################### DIAGNOSIS CODE for precondition_RWM_multi ####################

# available options for result_to_diagnose:
# - results_model_1_multi_pcond
# - results_model_2_multi_pcond
# - results_model_3_multi_pcond
# - results_model_4_multi_pcond

# available options for y_true:
# - Y_logit
# - Y_cauchit

results_pcond <- results_model_1_multi_pcond

result_to_diagnose_pcond <- rbind(results_pcond[[1]]$x_store, results_pcond[[2]]$x_store, 
                            results_pcond[[3]]$x_store,results_pcond[[4]]$x_store,results_pcond[[5]]$x_store)

y_true <- Y_logit


# TRACE PLOT
par(mfrow=c(2, 5), mar=c(4, 4, 2, 1))  

for (i in 1:ncol(result_to_diagnose_pcond)) {
  
  #cumulative posterior means
  cumulative_means_pcond <- cumsum(result_to_diagnose_pcond[, i]) / seq_along(result_to_diagnose_pcond[, i])
  
  # adjust limit
  data_range_pcond <- range(result_to_diagnose_pcond[, i], cumulative_means_pcond, beta[i])
  
  ylim_pcond <- c(min(data_range_pcond) - 0.1 * diff(data_range_pcond), max(data_range_pcond) + 0.1 * diff(data_range_pcond))
  
  # main plot
  plot(result_to_diagnose_pcond[, i], type = 'l', col = rgb(0.5, 0.5, 0.5, 0.5), lwd = 0.5,
       main = paste("Beta", i), xlab = "Iteration", ylab = "Value", ylim = ylim_pcond)
  
  # plot cumulative posterior means
  lines(cumulative_means_pcond, col = "blue", lwd = 2)  
  
  # true beta line
  abline(h = beta[i], col = "red", lwd = 2) 
}


# effective Sample Size
mcmc_samples_diagnose_pcond <- mcmc.list(mcmc(tail(result_to_diagnose_pcond, 10000)))
ess_diagnose_pcond <- effectiveSize(mcmc_samples_diagnose_pcond)
print(paste("Effective Sample Size: ",ess_diagnose_pcond))

# calculate Brier score
set.seed(0790)
p_forecast_diagnose_pcond <- plogis(X %*% colMeans(tail(result_to_diagnose_pcond, 10000)))
brier_score_diagnose_pcond <- mean((y_true - p_forecast_diagnose_pcond)^2)
print(paste("Brier score Model 1: ",brier_score_diagnose_pcond))

# report mean and CI
posterior_samples <- results_pcond[[5]]$x_store

posterior_means <- apply(posterior_samples, 2, mean)
lower_quantiles <- apply(posterior_samples, 2, quantile, probs = 0.025)
upper_quantiles <- apply(posterior_samples, 2, quantile, probs = 0.975)

print("Posterior Means for each parameter:")
print(posterior_means)
print("2.5% Quantiles for each parameter:")
print(lower_quantiles)
print("97.5% Quantiles for each parameter:")
print(upper_quantiles)





