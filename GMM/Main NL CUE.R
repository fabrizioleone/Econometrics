# Monte Carlo Simulation - Estimate Non-linear model via continuously updating GMM
# Fabrizio Leone
# 19 - 04 - 2021

# Call packages ----
  rm(list = ls(all=TRUE))
  if (!require("pacman")) install.packages("pacman")
  pacman::p_load(data.table,  tictoc)
  set.seed(123)
  #dev.off()

# Define Parameters ----
  Nobs <- 1000
  beta <- c(0.2, -0.1, 0.3)
  rep  <- 1000

# Define GMM objective function ----
  objGMM <- function(beta, df){
    
    # Get error term
      u          <- df$y - beta[1] - beta[2]*log(df$X / (df$Z*beta[3]))
    
    # Compute moment conditions
      m0         <- sum(u)
      m1         <- crossprod(df$Z, u)
      m2         <- crossprod(df$X, u)
      m          <- c(m0, m1, m2)
      
    # Compute opt VC matrix
      mvc0       <- u
      mvc1       <- df$Z * u
      mvc2       <- df$X * u
      mvc        <- cbind(mvc0, mvc1, mvc2)
      
    # Form obj function (with continuously updating weighting matrix)
      V          <- solve(crossprod(mvc, mvc))   # inverse of moment outer product
      g          <- crossprod(m, V) %*% m        # t(m) %*% V %*% m
      g          <- g / dim(df)[1]
      
    # Return obj fun
    return(g)
    
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
      dt[, y  := beta[1] + beta[2]*log(X / (Z*beta[3])) + e]
      
    # 2. Run optimization 
      res   <- optim(par     = c(0.15, -0.05, 0.25),
                     fn      = objGMM,
                     method  = "Nelder-Mead",
                     df      = dt)$par
      
    
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


