#------------- Objective function ------------#

Obj_function <- function(x0, X, A, price, share, v, nn,  Z, sharesum, W, prods, IDprod, IDmkt){
 
  # Initialize parameters
  theta1      <- x0[1:5]
  theta2      <- x0[6:9]              
  ii          <- 0
  norm_max    <- 1
  delta       <- as.matrix(X)%*%theta1
  tol_inner   <- 1.e-14                                          # Tolerance for inner loop (NFXP)
  TM          <- 50

  while (norm_max > tol_inner  && ii<1000) {        
    
    #step 1: simulated market share 
    num       <- delta %*% matrix(1,1,nn)*exp(as.matrix(cbind(A, price))%*%as.matrix((theta2*v)))
    den       <- 1+sharesum%*%num
    den       <- t(sharesum)%*%den
    sim_share <- rowMeans(num/den)
    
    
    #step 2: compute a new delta by BLP and compute norm_max
    delta_new <- delta*(share/sim_share)                       # BLP contraction mapping
    norm_max  <- max(abs(delta_new - delta))        
    delta     <- delta_new       
    ii        <- ii+1
    
  }
  
  
  #step 3: get the implied structural errors 
  xi          <- log(delta_new) - as.matrix(X)%*%theta1          # Updated structural error  
  g           <- t(Z)%*%xi;                                      # Moment conditions GMM
  
  
  #step 4: update GMM objective function
  f           <- t(g)%*%W%*%g;  
  
  
  #step 5: gradient
  attr(f, "gradient")  <- Gr_function(num, den, prods, IDmkt, IDprod, X, v, Z, W, xi)
  
  #return(list(f,gradient))
  
  return(f)

  
  
  
}










