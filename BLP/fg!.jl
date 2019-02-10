function fg!(F, G, x)
    tol_inner  = 1.e-14;                                 # Tolerance for inner loop (NFXP)
    theta1     = x[1:5];                                 # Linear parameters
    theta2     = x[6:9];                                 # Non Linear Paramters
    ii         = 0;
    norm_max   = 1;
    delta      = X*theta1;

    while norm_max > tol_inner  && ii < 1000

         # Step 1: Simulated market shares
         num       = delta.*exp.([A price]*(theta2.*v)); # Numerator of simulated integral
         den       = ones(TM,1).+sharesum*num;           # Denominator of simulated integral
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
         f         = g'*W*g;

    #------------- Specify Gradient------------#
    if !(G==nothing)
         d1            = zeros(25,25,50);
         d2            = zeros(970,4);
         D1            = zeros(970,4);
         Grad_fun      = zeros(970,9);

    # 1. Compute Jacobian Matrix

    # partial share\ partial theta2
    for m = 1:TM
        for p = 1:prods[m]
            for pp = 1:prods[m]
            if p == pp
                d1[pp,p,m] = mean(share[(IDmkt.==m) .& (IDprod.==p),:].*(ones(1,size(v)[2]).-share[(IDmkt.==m) .& (IDprod.==p),:]));
            else
                d1[pp,p,m] = -mean(share[(IDmkt.==m) .& (IDprod.==p),:].*(share[(IDmkt.==m) .& (IDprod.==pp),:]));
            end
            end
        end
    end

    # partial share\ partial sigma
    for j = 2:size(X,2)
        d2[:,j-1] = mean(v[j-1,:]'.*share.*(X[:,j] .- sharesum'*(sharesum*(X[:,j].*share))),dims=2);
    end

    for m = 1:TM
        D1[T[m,1]:T[m,2],:] = -d1[1:prods[m],1:prods[m],m]\d2[T[m,1]:T[m,2],:];
    end

    Grad_fun[:,6:9] = D1;
    Grad_fun[:,1:5] = -X;

    # 1. Compute gradient
    grad       = 2*Grad_fun'*Z*W*Z'*xi;
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

    #------------- Specify Objective Function ------------#
    if !(F == nothing)
        f         = tr(f);                             # take trace of f to ensure it is Float64
        return f
    end

end
