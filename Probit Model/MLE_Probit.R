# Monte Carlo Simulation - Probit Model
# Fabrizio Leone
# 06 - 02 - 2019

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

## Define Parameters
N    <- 1000
beta <- c(0.2,-0.1)
rep  <- 1000

# Define probit objective function
probit_obj <- function(beta, y, X) {
  cdf  <- pnorm(X%*%beta,0,1)
  l    <- y*log(cdf) + (1-y)*log((1 - cdf))  # log-likelihood
  return( if((-mean(l))==Inf | is.na(-mean(l))){  10e5 } else{ -mean(l) } )}

# Define gradient function
probit_gr <- function(beta, y, X) {
  cdf  <- pnorm(X%*%beta,0,1)
  pdf  <- dnorm(X%*%beta,0,1)
  v    <- cbind(pdf*X[,1]*(y-cdf),pdf*X[,2]*(y-cdf))
  u    <- cbind(cdf*(1-cdf),cdf*(1-cdf))
  s    <- v/u
  return( -colMeans(s) )}


# Run simulation
tic()

results <- do.call(rbind, mclapply(1:rep, function(i){
  
  # 1. Simulate Data
  const   <- matrix(1,N)
  X       <- cbind(const, rchisq(N, 10, ncp = 0))
  epsilon <- rnorm(N,0,1)
  y       <- as.numeric(X%*%beta > epsilon)
  
  # 2. Run optimization 
  res     <- optim(c(0,0), probit_obj, probit_gr, method ="BFGS", hessian=TRUE, X=X, y=y)$par
  
}))

#  Show and Plot results
colMeans(results)
std.error(results)
plot(density(results[,1]))
plot(density(results[,2]))

toc()

## Problems of the code:
# 1. very sensitive to initial values -> check for a better solver

