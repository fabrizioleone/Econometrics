# Monte Carlo Simulation - Ordered Probit (Two Thresholds)
# Fabrizio Leone
# 07 - 02 - 2019


# Import Packages

#import Pkg; Pkg.add("Distributions")
#import Pkg; Pkg.add("LinearAlgebra")
#import Pkg; Pkg.add("Optim")
#import Pkg; Pkg.add("NLSolversBase")
#import Pkg; Pkg.add("Random")
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("Statistics")

cd("$(homedir())/Documents/GitHub/Econometrics")

using Distributions, LinearAlgebra, Optim, NLSolversBase, Random, Plots, Statistics
Random.seed!(10);

# Define Parameters
N           = 1000;
beta        = [-0.1; 0.2];                                                      # Coefficients
alpha       = [-1; 0.5];                                                        # Thresholds
Npar        = length(alpha)+length(beta);
x0          = rand(Normal(0,1),Npar,1);                                         # Starting values
repetitions = 1000;

# Define ordered probit objective function
function nll_OrderedProbit(pars::Array{Float64,1}, y::Array{Int64,2}, x::Array{Int64,2})
    thresholds  = [-Inf pars[3] pars[4] Inf];
    Xb          = x[:,1].*pars[1] + x[:,2].*pars[2];
    p           = cdf.(Normal(0,1),thresholds[y.+1] - Xb) - cdf.(Normal(0,1),thresholds[y] - Xb);
    nll         = - mean(log.(p));
    return nll
end


# Run Monte Carlo Simulation
beta_hat = zeros(N,Npar);

@time begin

for i = 1:repetitions

# 1. Simulate Data
W         = rand(Poisson(3),N,2);
ϵ         = rand(Normal(0,1),N,1);
ystar     = W[:,1].*beta[1] + W[:,2].*beta[2] + ϵ
y         = Int.(1 .+ (ystar.>alpha[1]) + (ystar.>alpha[2]));

# 2. Run optimization
function objfun(b)
        nll_OrderedProbit(b[1:4],y,W)
end

res  = Optim.optimize(objfun,x0)
beta_hat[i,:]  = res.minimizer

# alternative method: not working
#ODJ            = OnceDifferentiable(only_fg!(fg!), x0)
#NLSolversBase.value_gradient!(ODJ, x0)
#gradient(ODJ)
#@time beta_hat[i,:]    = Optim.optimize(ODJ,startvalues)
#res           = Optim.optimize(ODJ, x0, LBFGS())
#beta_hat[i,:] = res.minimizer


end

end


# Show and Plot results
println("Mean Estimates ",sum!([1. 1. 1. 1.], beta_hat)/repetitions)
println("SE Estimates ", std(beta_hat; mean=nothing, dims=1))
h1 = histogram(beta_hat[:,1], label = "beta_1")
h2 = histogram(beta_hat[:,2], label = "beta_2")
h3 = histogram(beta_hat[:,3], label = "alpha_1")
h4 = histogram(beta_hat[:,4], label = "alpha_2")
plot(h1,h2,h3,h4, layout=(2,2),legend=false)
