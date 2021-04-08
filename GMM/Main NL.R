# Monte Carlo Simulation - Estimate Non-linear model via GMM
# Fabrizio Leone
# 08 - 04 - 2021

# Call packages ----
  rm(list = ls(all=TRUE))
  if (!require("pacman")) install.packages("pacman")
  pacman::p_load(data.table, gmm, tictoc)
  set.seed(123)
  #dev.off()

# Define Parameters ----
  Nobs <- 1000
  beta <- c(0.2,-0.1,0.3)
  rep  <- 1000

# Define GMM objective function ----
  objGMM <- function(beta, df){
    
    # Create moment conditions
    u <- df$y - beta[1] - beta[2]*df$X - log(df$Z*beta[3])
    m0 <- df$Z * u
    m1 <- df$X * u
    m2 <- u
    m  <- cbind(m0, m1, m2)
    return(m)
  }


# Run simulation ----
  tic()
  
  results   <- do.call(rbind, lapply(1:rep, function(i){
    
    # 1. Simulate Data
    dt   <- data.table()
    dt[, ID := 1:Nobs]
    dt[, X  := rchisq(.N, 3)]
    dt[, Z  := rchisq(.N, 4)]
    dt[, e  := rnorm(.N)]
    dt[, y  := beta[1] + X*beta[2] + log(Z*beta[3]) + e]
    
    # 2. Run optimization 
    res   <- gmm(objGMM, dt, c(0,0,0.1))$coefficients
    
  }))

# Show the results ---
  par(mfrow=c(1,3))
  
  d <- density(results[, 1])  
  plot(d,
       frame = FALSE,
       col = "steelblue", 
       main = " ",
       xlab = "Intercept",
       xlim = c(min(d$x) - 0.1, max(d$x) + 0.1),
       ylim = c(min(d$y), max(d$y) + 0.1))
  
  d <- density(results[, 2])  
  plot(d,
       frame = FALSE,
       col = "steelblue", 
       main = " ",
       xlab = "Slope",
       xlim = c(min(d$x) - 0.1, max(d$x) + 0.1),
       ylim = c(min(d$y), max(d$y) + 0.1))  
  
  d <- density(results[, 3])  
  plot(d,
       frame = FALSE,
       col = "steelblue", 
       main = " ",
       xlab = "Slope",
       xlim = c(min(d$x) - 0.1, max(d$x) + 0.1),
       ylim = c(min(d$y), max(d$y) + 0.1))  
  
  
  # Table
  t <- rbind(apply(results, 2, mean), apply(results, 2, sd))
  print(t)

toc()


