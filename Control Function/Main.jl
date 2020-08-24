# Monte Carlo Simulation for control function approach with high dimensiona fixed effects
# Fabrizio Leone - 2020
# fabrizioeone93@gmail.com

# ------------------------------ Initialization ------------------------------ #
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("FixedEffectModels")
#Pkg.add("Random")
#Pkg.add("Distributions")
#Pkg.add("LinearAlgebra")
#Pkg.add("Distributed")
#Pkg.add("Plots")
using CSV, DataFrames, FixedEffectModels, Random, LinearAlgebra, Distributions, Distributed, Base.Threads, Plots
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
    aID  = sum(kron(rand(Normal(6,1),ctr.N,1),Matrix{Int64}(I,ctr.T,ctr.T)),dims=2)  # individual FE
    aT   = sum(kron(Matrix{Int64}(I,ctr.N,ctr.N),rand(Normal(5,10),ctr.T,1)),dims=2) # time FE
    v    = rand(Normal(0,1),ctr.N*ctr.T,1)                                           # endogenous part of X
    u    = ctr.rho*v .+ rand(Normal(0,1),ctr.N*ctr.T,1)                              # error main regressions
    Z    = exp.(rand(Normal(0,1),ctr.N*ctr.T,1))                                     # Instrument for X
    X    = aID .+ aT .+ par.gamma1*Z .+ v                                            # endogenous regressor
    y    = aID .+ aT .+ X * par.beta1  .+ u                                          # outcome
    df   = DataFrame([ID Time y X Z])
    colnames = ["ID","Time","y","X","Z"]
    rename!(df, colnames)

    return df
end

# Create function to run Monte Carlo Simulation
function MonteCarlo(par,ctr)
    # Crate data
    df = Create_data(par,ctr)

    # 1.Endogenous Regression
    out1 = reg(df, @formula(y ~ X + fe(ID) + fe(Time))).coef[1]

    # 2. Control Function Approach
    df[!, :CF] = StatsBase.residuals(reg(df, @formula(X ~ Z + fe(ID) + fe(Time)), save = true))
    out2 = reg(df, @formula(y ~ X + CF + fe(ID) + fe(Time))).coef[1]

    return [out1 out2]
end

# Initialize function to run the simulation
function MC_execute(par,ctr)
    out = zeros(ctr.Boot,2)
        @threads for i = 1:ctr.Boot
                    out[i,:] = MonteCarlo(par,ctr)
                 end

    return out
end

#Run simulation
par = Params(1.0, 2.0)
ctr = Ctr(100, 5, 1.5, 1000)
@time MCout = MC_execute(par,ctr)
@show mean(MCout,dims=1) std(MCout,dims=1)

# Plot results
h1 = histogram(MCout[:,1])
h2 = histogram(MCout[:,2])
plot(h1,h2,layout=(1,2),legend=false)
