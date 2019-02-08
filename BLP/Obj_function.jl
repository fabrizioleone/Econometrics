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

     # step 1: simulated market share
     num       = delta.*exp.([A price]*(theta2.*v));
     den       = ones(TM,1).+sharesum*num;
     den       = sharesum'*den;
     sim_share = (1/nn)*sum(num./den,2);


global i += 1
end
