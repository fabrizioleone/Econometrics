# Monte Carlo Simulation - Logit Model 
# Fabrizio Leone
# 05 - 02 - 2019

## Housekeeping
rm(list = ls(all=TRUE))
cat("\f")
#dev.off()
clr <- function(){cat(rep("\n", 50))}   
set.seed(1)

## Call packages
Packages <- c("nloptr", "evd", "tictoc", "parallel", "plotrix")
invisible(lapply(Packages, library, character.only = TRUE))
#invisible(lapply(Packages, install.packages, character.only = TRUE)) # if some package is missing

## Initialization
N    <- 1000
beta <- c(0.2,-0.1)
opts <- list("algorithm"="NLOPT_LN_COBYLA","xtol_rel"=1.0e-8, "maxeval"=10e8)
rep  <- 1000

# Logit obj. function
logit_obj <- function(beta, y, X) {
             prob <- exp (X%*%beta ) /(1+ exp(X%*%beta ))
             l    <- log(y*prob + (1-y)*(1 - prob))  # log-likelihood
             return( if((-mean(l))==Inf | is.na(-mean(l))){  10e5 } else{ -mean(l) } ) }

# Run simulation
tic()

results <- do.call(rbind, mclapply(1:rep, function(i){
  
  # 1. simulate data
  const   <- matrix(1,N)
  X       <- cbind(const, rchisq(N, 10, ncp = 0))
  epsilon <- -rlogis(N,0,1)
  y       <- as.numeric(X%*%beta > epsilon)
  # 2. compute solution
  res     <- optim(c(1,-2), logit_obj, gr = NULL, method ="BFGS", hessian=TRUE, X=X, y=y)$par
  
}))

# Display Output
colMeans(results)
std.error(results)
plot(density(results[,1]))
plot(density(results[,2]))

toc()

## Problems of the code:
# 1. very sensitive to initial values -> check for a better solver
# 2. cannot introduce gradient function

