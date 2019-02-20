# Monte Carlo Simulation - Ordered Probit Model (Two Thresholds)
# Fabrizio Leone
# 20 - 02 - 2019

## Housekeeping
rm(list = ls(all=TRUE))
cat("\f")
#dev.off()
clr <- function(){cat(rep("\n", 50))}   
set.seed(10)

## Call packages
Packages <- c("evd", "tictoc", "parallel","plotrix","numDeriv")
invisible(lapply(Packages, library, character.only = TRUE))
#invisible(lapply(Packages, install.packages, character.only = TRUE)) # uncomment if some package is missing

## Define Parameters
N            <- 1000
beta         <- c(-0.1, 0.2)
alpha        <- c(-1, 0.5)
Npar         <- length(alpha) + length(beta)
startvalues  <- rnorm(Npar,0,1)
repetitions  <- 1000

  
# Define Ordered Probit objective function
oprobit_obj  <- function(pars, y, x) {
                thresholds  <- c(-Inf, pars[3], pars[4], Inf)
                Xb          <- x[,1]*pars[1] + x[,2]*pars[2]
                p           <- pnorm((t(thresholds[y+1])-Xb)) - pnorm((t(thresholds[y]-Xb)))
                nll         <- -mean(log(p))
                return( if(nll==Inf | is.na(nll)){  10e5 } else{ nll } )}


# Define Ordered Probit gradient function
oprobit_gr  <- function(pars, y, x) {
               thresholds   <- c(-Inf, pars[3], pars[4], Inf)
               Xb           <- x[,1]*pars[1] + x[,2]*pars[2]
               p            <- pnorm((t(thresholds[y+1])-Xb)) - pnorm((t(thresholds[y]-Xb)))
               dLdbeta1     <- (y==1)*dnorm(pars[3] - Xb)*(-x[,1])/p
                              +(y==2)*(dnorm(pars[4] - Xb) - dnorm(pars[3] - Xb))*(-x[,1])/p
                              +(y==3)*dnorm(pars[4] - Xb)*x[,1]/p
               dLdbeta2     <- (y==1)*dnorm(pars[3] - Xb)*(-x[,2])/p
                              +(y==2)*(dnorm(pars[4] - Xb) - dnorm(pars[3] - Xb))*(-x[,2])/p
                              +(y==3)*dnorm(pars[4] - Xb)*x[,2]/p
               dLdalpha1    <- (y==1)*dnorm(pars[3] - Xb)/p
                              +(y==2)*dnorm(pars[3] - Xb)*(-1)/p
               dLdalpha2    <- (y==2)*dnorm(pars[4] - Xb)/p
                              +(y==3)*dnorm(pars[4] - Xb)*(-1)/p
               gradient     <- rbind(dLdbeta1, dLdbeta2, dLdalpha1, dLdalpha2)
               ns           <- -rowMeans(gradient)
               
               return( ns )}



# Run simulation
tic()

results     <- do.call(rbind, mclapply(1:repetitions, function(i){
  
  # 1. Simulate Data
  x         <- cbind(rpois(N,3),rpois(N,3))
  epsilon   <- rnorm(N,0,1)
  ystar     <- x[,1]*beta[1] + x[,2]*beta[2] + epsilon
  y         <- 1 + as.numeric(ystar>alpha[1]) + as.numeric(ystar>alpha[2])
  
  
  # 2. Run optimization 
  res       <- optim(startvalues, oprobit_obj, oprobit_gr, method ="BFGS", hessian=TRUE, x=x, y=y)$par
  # gradient not working properly
  
}))

clr()

toc()


#  Show and Plot results
colMeans(results)
std.error(results)
plot(density(results[,1]))
plot(density(results[,2]))
plot(density(results[,3]))
plot(density(results[,4]))











