## BLP - Main
# Fabrizio Leone
# 07 - 02 - 2019

#------------- Install and Upload Packages -------------#
#import Pkg; Pkg.add("Distributions")
#import Pkg; Pkg.add("LinearAlgebra")
#import Pkg; Pkg.add("Optim")
#import Pkg; Pkg.add("NLSolversBase")
#import Pkg; Pkg.add("Random")
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("Statistics")
#import Pkg; Pkg.add("DataFrames")
#import Pkg; Pkg.add("CSV")
#import Pkg; Pkg.add("RecursiveArrayTools")

cd("$(homedir())/Documents/GitHub/Econometrics/BLP")

using Distributions, LinearAlgebra, Optim, NLSolversBase, Random, Plots, Statistics, DataFrames, CSV, RecursiveArrayTools
Random.seed!(10);

#------------- Read data and initialize useful objects -------------#
data           = CSV.read("data.csv", header=0, normalizenames=true)
IDmkt          = Vector{Int64}(data[:,1]);                                     # Market identifier
IDprod         = Vector{Int64}(data[:,2]);                                     # Product identifier
share          = Vector{Float64}(data[:,3]);                                   # Market share
A              = Matrix{Float64}(data[:,4:6]);                                 # Product characteristics
price          = Vector{Float64}(data[:,7]);                                   # Price
z              = Matrix{Float64}(data[:,8:10]);                                # Instruments
TM             = maximum(IDmkt);                                               # Number of markets
prods          = Vector{Int64}(zeros(TM));                                     # Number of products in each market
for m=1:TM
    prods[m,1] = maximum(IDprod[IDmkt.==m,1]);
end
T              = Matrix{Int64}(zeros(TM,2));
T[1,1]         = 1;
T[1,2]         = prods[1,1];
for i=2:TM
    T[i,1]     = T[i-1,2]+1;                                                   # 1st Column market starting point
    T[i,2]     = T[i,1]+prods[i,1]-1;                                          # 2nd Column market ending point
end
Total          = T[TM,2];                                                      # Number of obsevations
TotalProd      = maximum(prods);                                               # Max # of products in a given market
sharesum       = zeros(TM,Total);                                              # Used to create denominators in predicted shares (i.e. sums numerators)
denomexpand    = zeros(Total,1);                                               # Used to create denominators in predicted shares (expands sum numerators)
for i=1:TM
    sharesum[i,T[i,1]:T[i,2]]    = Matrix{Int64}(ones(1,prods[i]));
    denomexpand[T[i,1]:T[i,2],1] = i.*Matrix{Int64}(ones(prods[i],1));
end

#------------- Initialize Optimization -------------#
Kbeta          = 2+size(A,2);                                                  #  Number of parameters in mean utility
Ktheta         = 1+size(A,2);                                                  #  Number of parameters with random coefficient
nn             = 200;                                                          #  Draws to simulate shares
v              = rand(Normal(0,1),Ktheta,nn);                                  #  Draws for share integrals during estimation (we draw a fictious sample of 100 individuals from a normal distribution)
X              = [ones(size(A,1),1) A price];                                  #  Covariates
Z              = [ones(Total,1) A z A.^2 z.^2];                                #  Instruments
nZ             = size(Z,2);                                                    #  Number of instrumental variables
W              = inv(Z'*Z);                                                    #  Starting GMM weighting matrix
true_vals      = Array{Float64,2}([3 3 0.5 0.5 -2 0.8 0.5 0.5 0.5]');          #  True values used to generate the data
x0             = Array{Float64,2}([-0.5 -1 2 1 3 1.2 3 1 0.01]');              #  Random starting values
x_L            = [-Inf*ones(Kbeta,1);zeros(Ktheta,1)];                         #  Lower bounds is zero for standard deviations of random coefficients
x_U            = Inf.*ones(Kbeta+Ktheta,1);                                    #  Upper bounds for standard deviations of random coefficients

#------------- Run Optimization - 1st Stage-------------#
#function fun(x)
#    Obj_function(x[1:9],X,A,price,v,TM,sharesum,share,Z,W,IDmkt,IDprod)[1]
#end

#function gf!(G,x)
#    grad       = Obj_function(x[1:9],X,A,price,v,TM,sharesum,share,Z,W,IDmkt,IDprod)[2]
#    G[1]       = grad[1]
#    G[2]       = grad[2]
#    G[3]       = grad[3]
#    G[4]       = grad[4]
#    G[5]       = grad[5]
#    G[6]       = grad[6]
#    G[7]       = grad[7]
#    G[8]       = grad[8]
#    G[9]       = grad[9]
#end

### TRIAL SECTION

#ODJ = OnceDifferentiable(fun, gf!, x0)
#NLSolversBase.value_gradient!(ODJ, x0)
#gradient(ODJ)

#@time res      = optimize(ODJ, x0, BFGS())
#@time res      = optimize(fun,gf! x0, BFGS())
#[ true_vals res.minimizer ]


## trial 2

function fg!(F, G, x)
    tol_inner  = 1.e-14;                                 # Tolerance for inner loop (NFXP)
    theta1     = x[1:5];                                # Linear parameters
    theta2     = x[6:9];                                # Non Linear Paramters
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


    #------------- Specify Gradient------------#
if !(G==nothing)
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
                d1[pp,p,m] = mean(sim_share_ijm[(IDmkt.==m) .& (IDprod.==p),:].*(ones(1,size(v)[2]).-sim_share_ijm[(IDmkt.==m) .& (IDprod.==p),:]));
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
    grad     = 2*Grad_fun'*Z*W*Z'*xi;
    G[1]       = grad[1]
    G[2]       = grad[2]
    G[3]       = grad[3]
    G[4]       = grad[4]
    G[5]       = grad[5]
    G[6]       = grad[6]
    G[7]       = grad[7]
    G[8]       = grad[8]
    G[9]       = grad[9]
end


    if !(F == nothing)
        f         = tr(g'*W*g);                             # take trace of f to ensure it is Float64
        return f
    end

end

ODJ = OnceDifferentiable(only_fg!(fg!), x0)
NLSolversBase.value_gradient!(ODJ, x0)
gradient(ODJ)
@time res1 = Optim.optimize(ODJ,x_L,x_U,x0)

#------------- Obtain standard errors-------------#


 # To be done:
 # 1. understand why the code is so slow: check how fun and gf! are called
 # 2. check if matrix multiplications in obj_function takes much time and memory
 # 3. check why estimates are not precise

 res1.minimizer
