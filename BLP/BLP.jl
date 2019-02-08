## BLP
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

cd("$(homedir())/Documents/GitHub/Econometrics/BLP")

using Distributions, LinearAlgebra, Optim, NLSolversBase, Random, Plots, Statistics, DataFrames, CSV
Random.seed!(10);

#------------- Read data and initialize useful objects -------------#
data = CSV.read("data.csv")
IDmkt          = data[:,1];                       # Market identifier
IDprod         = data[:,2];                       # Product identifier
share          = data[:,3];                       # Market share
A              = data[:,4:6];                     # Product characteristics
price          = data[:,7];                       # Price
z              = data[:,8:10];                    # Instruments
TM             = maximum(IDmkt);                  # Number of markets
prods          = zeros(TM);                       # Number of products in each market
for m=1:TM
    prods[m,1] = maximum(IDprod[IDmkt.==m,1]);
end
T              = zeros(TM,2);
T[1,1]         = 1;
T[1,2]         = prods[1,1];
for i=2:TM
    T[i,1]     = T[i-1,2]+1;                      # 1st Column market starting point
    T[i,2]     = T[i,1]+prods[i,1]-1;             # 2nd Column market ending point
end
Total          = trunc(Int64, T[TM,2]);           # Number of obsevations
TotalProd      = maximum(prods);                  # Max # of products in a given market
sharesum_0     = zeros(TM,Total);                 # Used to create denominators in predicted shares (i.e. sums numerators)
denomexpand_0  = zeros(Total,1);                  # Used to create denominators in predicted shares (expands sum numerators)
T              = convert(Array{Int64,2}, T);      # Convert to Int64
prods          = convert(Array{Int64,1}, prods)   # Convert to Int64
for i=1:TM
    sharesum_0[i,T[i,1]:T[i,2]]    = convert(Array{Int64,2},ones(1,prods[i]));
    denomexpand_0[T[i,1]:T[i,2],1] = i.*convert(Array{Int64,2},ones(prods[i],1));
end
sharesum_0     = convert(Array{Int64,2}, sharesum_0);
denomexpand_0  = convert(Array{Int64,2}, denomexpand_0);

#------------- Initialize Optimization -------------#
