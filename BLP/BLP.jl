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

#----------- Run Optimization - 1st Stage ----------#
ODJ            = OnceDifferentiable(only_fg!(fg!), x0)
#NLSolversBase.value_gradient!(ODJ, x0)
#gradient(ODJ)
@time res1     = Optim.optimize(ODJ,x_L,x_U,x0)

#------------- Obtain standard errors -------------#


 # To be done:
 # 1. understand why the code is so slow: check how fun and gf! are called
 # 2. check if matrix multiplications in obj_function takes much time and memory
 # 3. check why estimates are not precise

 res1.minimizer

 
