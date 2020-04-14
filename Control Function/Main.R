# ------------- Initialization ------------- #

# Monte Carlo Simulation for control function approach with high dimensiona fixed effects
# Fabrizio Leone - 2020
# fabrizioeone93@gmail.com

# Session info
#install.packages("devtools")
#devtools::session_info()

# Call packages
rm(list = ls(all=TRUE))
Packages  <- c("data.table", "fixest", "tictoc")
#invisible(lapply(Packages, install.packages))                      # Uncomment to install packages
invisible(lapply(Packages, library, character.only = TRUE))
set.seed(12784)


# ------------- Setup ------------- #
rep.sim <- 100                                                      # No. Monte Carlo replications                             
rho     <- 0.5                                                      # Correlation between 1st and 2nd stage error
beta    <- c(1, 0.2)                                                # Params to estimate
gamma   <- c(1, -0.15)                                              # Parames first stage

# ------------- Simulation ------------- # 
tic()

res.out <- do.call(rbind, lapply(1:rep.sim, function(i){

# Draw data
df      <- data.table(expand.grid(i = 1:50, j = 1:20, t = 1:10))
df[, v  := rnorm(.N)]                                                # Endogenous part of X
df[, u  := rho*v + rnorm(.N)]                                        # Main regression error
df[, Z1 := exp(rnorm(.N))]                                           # Exogenous IV
df[, a1 := rnorm(1), by = i]                                         # i fixed effect
df[, a2 := rnorm(1), by = j]                                         # j fixed effect
df[, a3 := rnorm(1), by = t]                                         # t fixed effect
df[, X  := gamma[1] + gamma[2]*Z1 + 0.5*a1 + 0.2*a2 + 0.3*a3 + v]    # Endogenous variable
df[, y  := beta[1] + beta[2]*X + a1 + a2 + a3 + u]                   # Linear outcome
df[, y1 := rpois(.N, exp(beta[1] + beta[2]*X + a1 + a2 + a3 + u))]   # Non-linear outcome (Poisson)

# Estimate parameters 

# Stage 1. Regress X on instruments
out.1.A    <- feols(X ~ Z1 | i + j + t, df)                          # Linear first stage
df[, CF1   := out.1.A[["residuals"]]]                                # Get errors, i.e. "control function"

# Stage 2: Run control function 
out.2.A    <- feols( y  ~ X + CF1 | i + j + t, df)                   # Control function linear model
out.2.B    <- feglm( y1 ~ X + CF1 | i + j + t, df)                   # Control function non-linear model

# Store Results
c(out.2.A[["coefficients"]][1],out.2.B[["coefficients"]][1])
  

}))

toc()

# ------------- Results ------------- # 
res        <- data.table(res.out)
names(res) <- c("Linear Model", "Poisson Model")
quantile   <- t(sapply(res, quantile))
mean       <- sapply(res, mean)
SE         <- sapply(res, sd)
t.stat     <- (mean - beta[2]) / SE                                  # Estimated coeff vs. true value
true       <- rep(beta[2], ncol(res))
print(cbind(true, quantile, mean, SE, t.stat))


# Comment
# Control function delivers consistent estimates for the parameters of 
# interest, even with high dimensional FE, both in linears and non-linear models








