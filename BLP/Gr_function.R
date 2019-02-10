
Gr_function <- function(num, den, prods, IDmkt, IDprod, X, v, Z, W, xi)

sim_share_ijm <- num/den;
d1            <- array(NA,c(25,25,50));
d2            <- array(NA,c(970,4));
D1            <- array(NA,c(970,4));
TM            <- 50

# 1. Compute Jacobian Matrix

# partial share\ partial theta2
for (m in 1:TM) {
for (p in 1:prods[m] ) {
for (pp in 1:prods[m] ) {
if (p == pp) {
d1[pp,p,m] <- mean(sim_share_ijm[IDmkt==m & IDprod==p,]*(matrix(1,nn,1)-sim_share_ijm[IDmkt==m & IDprod==p,]))
} else {
  d1[pp,p,m] <- -mean(sim_share_ijm[(IDmkt==m) & (IDprod==p),]*(sim_share_ijm[(IDmkt==m) & (IDprod==pp),]))
} 
}
}
  }

# partial share\ partial sigma
for (j in  2:dim(X)[2]){
d2[,j-1] <- rowMeans(v[j-1,]*sim_share_ijm*(X[,j] - t(sharesum)%*%(sharesum%*%(X[,j]*sim_share_ijm))));
}

for (m in 1:TM){
D1[T[m,1]:T[m,2],] <- solve(-d1[1:prods[m],1:prods[m],m],d2[T[m,1]:T[m,2],])
}

Jacob <- cbind(-X,D1)

gradient        <- 2*as.matrix(t(Jacob))%*%as.matrix(Z)%*%W%*%t(Z)%*%xi

return(gradient)


