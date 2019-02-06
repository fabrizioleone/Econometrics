% Monte Carlo Simulation - Logit Model 
% Fabrizio Leone
% 06 - 02 - 2019

clear all
clc

% Define Paramters
N           = 1000;
beta        = [-0.2,-0.1]';
startvalues = [0,0]';
repetitions = 1000;  
options     = optimoptions('fminunc','Display','off','GradObj','on','MaxIter',100000);
rng(10);

% Preallocate matrices
betahat     = NaN(repetitions, 2);
nll         = NaN(repetitions,1);
ns          = NaN(repetitions,2);
nH          = NaN(repetitions,2,2);

tic

% Monte Carlo
 for i = 1: repetitions

% 1. Simulate Data
const      = ones(N,1);
x          = [const , chi2rnd(10,N,1)];
epsilon    = randn(N,1);
y          = x*beta > epsilon ;
objfun     = @(b) nll_probit (b,y,x); 

% 2. Run optimization 
[betahat(i,:),nll,~,~,ns(i,:),nH(i,:,:)] = fminunc (objfun , startvalues , options);
 end
 
 toc
 
% Show and Plot results

mean(betahat) 
std(betahat)
ksdensity(betahat(:,1))
ksdensity(betahat(:,2))
   
   
   
   