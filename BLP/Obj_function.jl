## BLP - Objective Function
# Fabrizio Leone
# 07 - 02 - 2019

function Obj_function(x0::Vector{Float64},X::Matrix{Float64},A::Matrix{Float64},
                     price::Vector{Float64},v::Matrix{Float64},TM::Int64,
                     sharesum::Matrix{Float64},share::Vector{Float64},
                     Z::Matrix{Float64},W::Matrix{Float64},IDmkt::Vector{Int64},
                     IDprod::Vector{Int64})

#------------- Initialize Parameters-------------#
tol_inner  = 1.e-14;                                 # Tolerance for inner loop (NFXP)
theta1     = x0[1:5];                                # Linear parameters
theta2     = x0[6:9];                                # Non Linear Paramters
ii         = 0;
norm_max   = 1;
delta      = X*theta1;

while norm_max > tol_inner  && ii < 1000

     # Step 1: Simulated market shares
     global num= delta.*exp.([A price]*(theta2.*v)); # Numerator of simulated integral
     global den= ones(TM,1).+sharesum*num;           # Denominator of simulated integral
     den       = sharesum'*den;                      # Denominator of simulated integral
     sim_share = mean(num./den,dims=2);              # Simulated shares

     # Step 2: Compute a new delta by BLP inversion and compute norm_max
     global delta_new = delta.*(share./sim_share);   # BLP contraction mapping
     norm_max  = maximum(abs.(delta_new - delta));   # Find maximum of Euclidean distance
     delta     = delta_new;                          # Update delta
     ii        += 1                                  # Update counter

end

     # Step 3: Get the implied structural errors
     xi        = log.(delta_new) - X*theta1;         # Updated moment condition
     g         = Z'*xi;                              # Moment conditions GMM

     # Step 4: Update GMM objective function
     f         = tr(g'*W*g);                         # Take trace to ensure f is Float64


#------------- Specify Gradient------------#

sim_share_ijm = num./den;
d1            = zeros(25,25,50);
d2            = zeros(970,4);
D1            = zeros(970,4);
Grad_fun      = zeros(970,9);

# 1. Compute Jacobian Matrix
for m = 1:TM
    for p = 1:prods[m]
        for pp = 1:prods[m]
        if p == pp
            d1[pp,p,m] = mean(sim_share_ijm[(IDmkt.==m) .& (IDprod.==p),:].*(ones(1,size(v)[2])-sim_share_ijm[(IDmkt.==m) .& (IDprod.==p),:]));
        else
            d1[pp,p,m] = -mean(sim_share_ijm[(IDmkt.==m) .& (IDprod.==p),:].*(sim_share_ijm[(IDmkt.==m) .& (IDprod.==pp),:]));
        end
        end
    end
end

for j = 2:size(X,2)
    d2[:,j-1] = mean(v[j-1,:]'.*sim_share_ijm.*(X[:,j] .- sharesum'*(sharesum*(X[:,j].*sim_share_ijm))),dims=2);
end

for m = 1:TM
    D1[T[m,1]:T[m,2],:] = -d1[1:prods[m],1:prods[m],m]\d2[T[m,1]:T[m,2],:];
end

Grad_fun[:,6:9] = D1;
Grad_fun[:,1:5] = -X;

# 2. Compute gradient
grad     = 2*Grad_fun'*Z*W*Z'*xi;

return f, gradf

end
