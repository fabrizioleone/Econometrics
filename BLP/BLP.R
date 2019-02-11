# BLP
## Fabrizio Leone
## 11 - 02 - 2019

#----------- Housekeeping ----------- #
rm(list = ls(all=TRUE))
cat("\f")
Packages <- c("SparseM", "tictoc","optimParallel", "optimx")
invisible(lapply(Packages, library, character.only = TRUE))
#dev.off()
set.seed(10)  
clr <- function(){cat(rep("\n", 50))}   
setwd("~/Documents/GitHub/Econometrics/BLP")
source("Obj_function.R")
source("Gr_function.R")

#----------- Import and Tidy Data ----------- #
DATA           <- read.csv("~/Documents/GitHub/Econometrics/BLP/data.csv", header=FALSE)
IDmkt          <- DATA[,1]                                 # Market identifier
IDprod         <- DATA[,2]                                 # Product identifier
share          <- DATA[,3]                                 # Market share
A              <- DATA[,4:6]                               # Product characteristics
price          <- DATA[,7]                                 # Price
z              <- DATA[,8:10]                              # Instruments
TM             <- max(IDmkt)                               # Number of markets
prods          <- rep(0,TM)                                # Number of products in each market
for (m in 1:TM){
prods[m]       <- max(IDprod[IDmkt==m])
}
T              <- matrix(0,TM,2)
T[1,1]         <- 1
T[1,2]         <- prods[1] 
for (i in 2:TM){
T[i,1]         <- T[i-1,2]+1                                  # 1st Column market starting point
T[i,2]         <- T[i,1]+prods[i]-1                           # 2nd Column market ending point
}
Total          <- T[TM,2]                                     # Number of obsevations
TotalProd      <- max(prods)                                  # Max number of products in a given market
# Sparse matrices
#sharesum1       <- as.matrix.csr(matrix(0, TM, Total));      # Used to create denominators in predicted shares (i.e. sums numerators)
#denomexpand1    <- as.matrix.csr(matrix(0, Total, 1))        # Used to create denominators in predicted shares (expands sum numerators)
sharesum        <- matrix(0, TM, Total);                      # Used to create denominators in predicted shares (i.e. sums numerators)
denomexpand     <- matrix(0, Total, 1);  
#object.size(sharesum1) object.size(sharesum)
for (i in 1:TM){
sharesum[i,T[i,1]:T[i,2]]    <- 1
denomexpand[T[i,1]:T[i,2],1] <- i
}

#----------- Initialize Optimization ----------- #
Kbeta           <- 2+dim(A)[2]                                # Number of parameters in mean utility
Ktheta          <- 1+dim(A)[2]                                # Number   of parameters with random coefficient 
nn              <- 200;                                       # Number of draws to simulate shares
v               <- t(matrix(rnorm(nn*4), ncol=Ktheta))        # Draws for share integrals during estimation (we draw a fictious sample of 100 individuals from a normal distribution)
X               <- cbind(matrix(1, Total, 1), A, price)       # Covariates
Z               <- cbind(matrix(1, Total, 1), A, z, A^2, z^2) # Instruments
nZ              <- dim(Z)[2]                                  # Number of instrumental variables
W               <- solve(t(Z)%*%as.matrix(Z))                 # Starting GMM weighting matrix
true_vals       <- c(3, 3, 0.5, 0.5, -2, 0.8, 0.5, 0.5, 0.5)  # True values used to generate data
x0              <- as.vector(matrix(rnorm(9),Ktheta+Kbeta))   # Random starting values
x_L             <- rbind(-Inf*matrix(1,Kbeta,1), matrix(0, Ktheta,1))    # Lower bounds is zero for standard deviations of random coefficients
x_U             <- Inf*matrix(1,Kbeta+Ktheta,1)    


#----------- Run Optimization (No gradient, about 1 hour) ----------- #
#tic()
#cl            <- makeCluster(detectCores()); setDefaultCluster(cl = cl)
#res           <- optimParallel(x0, Obj_function, gr = Gr_function, method = "L-BFGS-B", lower = x_L, upper = x_U,
#                              X        = X, 
#                              A        = A,
#                              price    = price,
#                              share    = share,
#                              v        = v,
#                              nn       = nn,
#                              Z        = Z,
#                              sharesum = sharesum,
#                              W        = W,
#                              prods    = prods,
#                              IDmkt    = IDmkt,
#                              IDprod   = IDprod)

#toc()

#----------- Run Optimization (With gradient) ----------- #
#tic()
res              <- nlminb(x0, Obj_function, gr = attr(Obj_function, "gradient"), lower = x_L, upper = x_U,
                           X        = X, 
                           A        = A,
                           price    = price,
                           share    = share,
                           v        = v,
                           nn       = nn,
                           Z        = Z,
                           sharesum = sharesum,
                           W        = W,
                           prods    = prods,
                           IDmkt    = IDmkt,
                           IDprod   = IDprod)

toc()
















