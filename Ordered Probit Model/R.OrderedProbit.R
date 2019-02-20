# Monte Carlo Simulation - Ordered Probit Model (Two Thresholds)
# Fabrizio Leone
# 20 - 02 - 2019

## Housekeeping
rm(list = ls(all=TRUE))
cat("\f")
#dev.off()
clr <- function(){cat(rep("\n", 50))}   
set.seed(1)

## Call packages
Packages <- c("evd", "tictoc", "parallel")
invisible(lapply(Packages, library, character.only = TRUE))

## Define Parameters
N            <- 1000
beta         <- c(-0.1, 0.2)
alpha        <- c(-1, 0.5)
Npar         <- length(alpha) + length(beta)
startvalues  <- rand
repetitions  <- 1000