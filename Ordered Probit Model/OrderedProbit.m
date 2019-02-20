% Monte Carlo Simulation - Ordered Probit Model (Two Thresholds)
% Fabrizio Leone
% 20 - 02 - 2019

clear all
close all
clc
rng(10)

% Define Parameters
N           = 1000;
beta        = [-0.1; 0.2];                                                 % Coefficients
alpha       = [-1; 0.5];                                                   % Thresholds
startvalues = rand(length(alpha)+length(beta),1);                          % Starting values
repetitions = 1000;
options     = optimoptions('fminunc','Display','off','GradObj','on');

% Preallocate matrices
thetahat    = NaN(repetitions, 4);
nll         = NaN(repetitions,1);
ns          = NaN(repetitions,2);
nH          = NaN(repetitions,4,4);

tic

% Monte Carlo
 for i = 1: repetitions

% 1. Simulate Data
x          = poissrnd(3,N,2);
epsilon    = normrnd(0,1,N,1);
ystar      = x(:,1).*beta(1)+x(:,2).*beta(2)+epsilon;
y          = 1+(ystar>alpha(1))+(ystar>alpha(2));
objfun     = @(b) nll_OrderedProbit(b,y,x); 

% 2. Run optimization 
[thetahat(i,:),nll,~,~,~,nH(i,:,:)] = fminunc(objfun, startvalues, options);

 end
 
 toc
 
% Show and Plot results

mean(thetahat) 
std(thetahat)

figure
title('Empirical Distribution of the Monte Carlo Estimators')
subplot(2,2,1)
ksdensity(thetahat(:,1))
xlabel('\beta_1')
subplot(2,2,2)
ksdensity(thetahat(:,2))
xlabel('\beta_2')
subplot(2,2,3)
ksdensity(thetahat(:,3))
xlabel('\alpha_1')
subplot(2,2,4)
ksdensity(thetahat(:,4))
xlabel('\alpha_2')
