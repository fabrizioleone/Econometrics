## BLP - Objective Function
# Fabrizio Leone
# 07 - 02 - 2019


#------------- Initialize Parameters-------------#
theta1     = x0[1:5];
theta2     = x0[6:9];
ii         = 0;
norm_max   = 1;
delta      = X*theta1;

while norm_max > tol_inner  && ii < 1000

     # Step 1: Simulated market shares
     num       = delta.*exp.([A price]*(theta2.*v)); # Numerator of simulated integral
     den       = ones(TM,1).+sharesum*num;           # Denominator of simulated integral
     den       = sharesum'*den;                      # Denominator of simulated integral
     sim_share = sum(num./den,dims=2);               # Simulated shares

     # Step 2: Compute a new delta by BLP inversion and compute norm_max
     delta_new = delta.*(share./sim_share);          # BLP contraction mapping
     norm_max  = maximum(abs.(delta_new - delta));   # Find maximum of Euclidean distance
     delta     = delta_new;                          # Update delta
     global i += 1                                   # Update counter

end

     # Step 3: Get the implied structural errors
     xi        = log(delta_new) - X*theta1;          # Updated moment condition
     g         = Z'*xi;                              # Moment conditions GMM

     # Step 4: Update GMM objective function
     f         = g'*W*g;      
