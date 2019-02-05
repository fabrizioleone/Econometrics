function [nll, ns] = nll_logit (beta ,y,X)

prob1      =exp (X* beta ) ./(1+ exp(X* beta )); 
l          =log(y.* prob1 +(1 -y).*(1 - prob1 ));      % likelihood
s          = (y.*(1 - prob1 ).*X -(1 -y).* prob1 .*X); % score

nll        = - mean (l); 
ns         = - mean (s); 