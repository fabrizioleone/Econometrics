function [nll, ns] = nll_probit (beta,y,X)

cdf     = normcdf(X*beta,0,1);
pdf     = normpdf(X*beta,0,1); 
l       = y.* log(cdf) +(1 - y).*log(1 - cdf); 
s       = (pdf.*X.*(y - cdf))./(cdf.*(1-cdf)); 
nll     = - mean (l); 
ns      = - mean (s);
end





