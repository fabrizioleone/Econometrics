# Monte Carlo Simulation for control function approach with high dimensiona fixed effects
# Fabrizio Leone - 2020
# fabrizioeone93@gmail.com

# ------------------------------ Initialization ------------------------------ #
#Pkg.add.(["DataFrames", "CSV", "FixedEffectModels", "GLFixedEffectModels", "GLM", "Random", "Distributions", "LinearAlgebra", "Distributed", "Plots"])
using CSV, DataFrames, GLM, GLFixedEffectModels, FixedEffectModels, Random, LinearAlgebra, Distributions, Distributed, Base.Threads, Plots
Random.seed!(1320);
rng = MersenneTwister(1320)

# Define structure with paramaters
struct Params
    gamma1::Float64
    beta1::Float64
end

struct Ctr
    N::Int64
    T::Int64
    rho::Float64
    Boot::Int64
end

# Define function to create the data
function Create_data(par,ctr)
    ID   = sum(kron(1:ctr.N,Matrix{Int64}(I,ctr.T,ctr.T)),dims=2)                    # Panel ID: individual
    Time = sum(kron(Matrix{Int64}(I,ctr.N,ctr.N),1:ctr.T),dims=2)                    # Panel ID: time
    aID  = sum(kron(rand(Normal(0,1),ctr.N,1),Matrix{Int64}(I,ctr.T,ctr.T)),dims=2)  # individual FE
    v    = rand(Normal(0,1),ctr.N*ctr.T,1)                                           # endogenous part of X
    u    = ctr.rho*v .+ rand(Normal(0,1),ctr.N*ctr.T,1)                              # error main regressions
    Z    = rand(Chisq(3),ctr.N*ctr.T,1)                                              # Instrument for X
    X    = aID .+ par.gamma1*Z .+ v                                                  # endogenous regressor
    y    = aID .+ X * par.beta1  .+ u                                                # outcome linear
    y1   = ones(ctr.T*ctr.N,1) .* NaN                                                # initialize outcome Poisson
    lbd  = exp.(aID  .+ X * par.beta1 .+ u)                                          # mean Poisson
    for j = 1:ctr.T*ctr.N
        y1[j] = rand(Poisson(lbd[j]))                                                # fill outcome Poisson
    end
    df   = DataFrame([ID Time y y1 X Z])
    colnames = ["ID","Time","y", "y1", "X","Z"]
    rename!(df, colnames)

    return df
end

# Create function to run Monte Carlo Simulation
function MonteCarlo(par,ctr)

    # Crate data
    df = Create_data(par,ctr)

    # 1.Endogenous Regression
    out1 = reg(df, @formula(y ~ X + FixedEffectModels.fe(ID))).coef[1]                                                       # Linear regression
    out2 = nlreg(df, @formula(y1 ~ X + GLFixedEffectModels.fe(ID)), Poisson(), LogLink(), start = [0.8]).coef[1]             # Poisson regression

    # 2. Control Function Approach
    df[!, :CF] = reg(df, @formula(X ~ Z + FixedEffectModels.fe(ID)), save = true).residuals
    out3 = reg(df, @formula(y ~ X + CF + FixedEffectModels.fe(ID))).coef[1]                                                  # Linear regression
    out4 = nlreg(df, @formula(y1 ~ X + CF + GLFixedEffectModels.fe(ID)), Poisson(), LogLink(), start = [0.8, 0.1]).coef[1]   # Poisson regression

    return [out1 out2 out3 out4]
end

# Initialize function to run the simulation
function MC_execute(par,ctr)
    out = ones(ctr.Boot,4) .* NaN
        @threads for i = 1:ctr.Boot
            try
                out[i,:] = MonteCarlo(par,ctr)
            catch
                skip     # Skip if nlreg fails to converge
            end
        end

    return out
end

# Run simulation
par = Params(0.5, 1.0);
ctr = Ctr(100, 5, 1.5, 1000);
@time MCout = MC_execute(par,ctr)

# Tabulate results
MCout   = DataFrame(MCout)
MCout   = filter(row -> ! isnan(row.x1), MCout)
MCout   = convert(Matrix, MCout);
res_out = DataFrame([mean(MCout, dims = 1); std(MCout, dims = 1)]);
colnames = ["mean - OLS","mean - PPML", "mean - OLS CF", "mean - PPML CF"];
rename!(res_out, colnames);
@show res_out
println("True mean is: ", par.beta1)

# Plot results
h1 = histogram(MCout[:,1]);
h2 = histogram(MCout[:,2]);
h3 = histogram(MCout[:,3]);
h4 = histogram(MCout[:,4]);
plot(h1,h2,h3,h4, layout=(2,2), legend=true)
