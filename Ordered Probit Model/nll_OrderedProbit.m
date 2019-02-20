function [nll, ns] = nll_OrderedProbit(pars,y,x)

 % 1. Objective Function
 thresholds  = [-Inf pars(3) pars(4) Inf];
 Xb          = x(:,1).*pars(1)+x(:,2).*pars(2);
 p           = normcdf((thresholds(y+1)'-Xb)) - normcdf((thresholds(y)'-Xb));
 nll         = - mean(log(p));

 % 2. Gradient
  dLdbeta1    = (y==1).*(normpdf(pars(3) - Xb).*(-x(:,1)))./p...
             + (y==2).*(normpdf(pars(4) - Xb) - normpdf(pars(3) - Xb)).*(-x(:,1))./p...
             + (y==3).*normpdf(pars(4) - Xb).*x(:,1)./p;
         
 dLdbeta2    = (y==1).*normpdf(pars(3) - Xb).*(-x(:,2))./p...
             + (y==2).*(normpdf(pars(4) - Xb) - normpdf(pars(3) - Xb)).*(-x(:,2))./p...
             + (y==3).*normpdf(pars(4) - Xb).*x(:,2)./p;
         
 dLdalpha1   = (y==1).*normpdf(pars(3) - Xb)./p...
             + (y==2).*normpdf(pars(3) - Xb).*(-1)./p;    
         
 dLdalpha2   = (y==2).*normpdf(pars(4) - Xb)./p...
             + (y==3).*normpdf(pars(4) - Xb).*(-1)./p;
 
 
 gradient    = [dLdbeta1, dLdbeta2, dLdalpha1, dLdalpha2];
 
 ns          = - mean(gradient); 
 

 
end