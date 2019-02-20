
function fg!(F, G, x::Array{Float64,2})
    thresholds  = [-Inf x[3] x[4] Inf];
    Xb          = W[:,1].*x[1] + W[:,2].*x[2];
    p           = cdf.(Normal(0,1),thresholds[y.+1] - Xb) - cdf.(Normal(0,1),thresholds[y] - Xb);


    if !(G == nothing)
    dLdbeta1    = (y.==1).*(pdf.(Normal(0,1), x[3] .- Xb).*(-W[:,1]))./p
                + (y.==2).*(pdf.(Normal(0,1), x[4] .- Xb) - pdf.(Normal(0,1), x[3] .- Xb)).*(-W[:,1])./p
                + (y.==3).*pdf.(Normal(0,1), x[4] .- Xb).*W[:,1]./p;

    dLdbeta2    = (y.==1).*(pdf.(Normal(0,1), x[3] .- Xb).*(-W[:,2]))./p
                + (y.==2).*(pdf.(Normal(0,1), x[4] .- Xb) - pdf.(Normal(0,1), x[3] .- Xb)).*(-W[:,2])./p
                + (y.==3).*pdf.(Normal(0,1), x[4] .- Xb).*W[:,2]./p;

    dLdalpha1   = (y.==1).*pdf.(Normal(0,1), x[3] .- Xb)./p
                + (y.==2).*pdf.(Normal(0,1), x[3] .- Xb).*(-1)./p;

    dLdalpha2   = (y.==2).*pdf.(Normal(0,1), x[4] .- Xb)./p
                + (y.==3).*pdf.(Normal(0,1), x[4] .- Xb).*(-1)./p;

    gradient    = hcat(dLdbeta1, dLdbeta2, dLdalpha1, dLdalpha2);
    ns          = - mean(gradient,dims=1);
    G[1]        = ns[1]
    G[2]        = ns[2]
    G[3]        = ns[3]
    G[4]        = ns[4]
    end

    if !(F == nothing)
        f         = - mean(log.(p));
        return f
    end

end
