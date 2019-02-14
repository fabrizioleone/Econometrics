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
Packages <- c("nloptr", "evd", "tictoc", "parallel", "plotrix", "GenSA")
invisible(lapply(Packages, library, character.only = TRUE))
#invisible(lapply(Packages, install.packages, character.only = TRUE)) # if some package is missing

## Define Parameters
N    <- 1000
beta <- c(0.2,-0.1)
rep  <- 1000

# Define logit objective function
logit_obj <- function(beta, y, X) {
             prob <- exp (X%*%beta ) /(1+ exp(X%*%beta ))
             l    <- log(y*prob + (1-y)*(1 - prob))  # log-likelihood
             return( if((-mean(l))==Inf | is.na(-mean(l))){  10e5 } else{ -mean(l) } )}

# Define gradient function
logit_gr <- function(beta, y, X) {
             prob <- exp (X%*%beta ) /(1+ exp(X%*%beta ))
             v    <- cbind(y*(1 - prob),y*(1 - prob))
             u    <- cbind((1 -y)* prob,(1 -y)* prob)
             s    <- v*X - u*X 
             return( -colMeans(s) )}



# Run simulation
tic()

results   <- do.call(rbind, mclapply(1:rep, function(i){
  
  # 1. Simulate Data
  const   <- matrix(1,N)
  X       <- cbind(const, rchisq(N, 10, ncp = 0))
  epsilon <- -rlogis(N,0,1)
  y       <- as.numeric(X%*%beta > epsilon)
  
  # 2. Run optimization 
  res     <- optim(c(0,0), logit_obj, logit_gr, method ="BFGS", hessian=TRUE, X=X, y=y)$par
  #res     <- GenSA(c(0,0), logit_obj, c(-Inf,-Inf), c(Inf,Inf),X=X, y=y)$par
}))

#  Show and Plot results
colMeans(results)
std.error(results)
plot(density(results[,1]))
plot(density(results[,2]))

toc()

## Problems of the code:
# 1. very sensitive to initial values -> check for a better solver

