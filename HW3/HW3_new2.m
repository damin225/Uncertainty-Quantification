%% Generate trainning data

nosample = 50;
E = zeros(nosample,4);
P = zeros(nosample,1);
for i=1:nosample
    E(i,:) = KLexpansion(30000,3000)';
    P(i) = randn(1)*400+2000;
end
trpt = [P E];
y = @(P,E1,E2,E3,E4) P.*(1./E1+1./E2+1./E3+1./E4);
g = y(trpt(:,1),trpt(:,2),trpt(:,3),trpt(:,4),trpt(:,5));

% normalize data
P = (P-2000)/400;
for i=1:4
    E(:,i) = (E(:,i)-30000)/3000;
end
trpt = [P E];

%% Build Gassuian Process Model

% Linear regression to get trend coefficients beta
F = [ones(nosample,1) trpt(:,1) (trpt(:,2)+trpt(:,3)+trpt(:,4)+trpt(:,5))];
beta = (F'*F)\(F'*g);

% Maximize log ML function to get parameters
Esum = sum(E,2);
trpt_new = [P,Esum];
lambda0=ones(2,1)*0.25;
f = @(lambda) logMLE2(trpt_new,nosample, g,F,beta, lambda);
options = optimoptions(@fminunc,'Display','iter','Algorithm',...
    'quasi-newton','OptimalityTolerance',1e-6,'StepTolerance',1e-6);
[lambda,logProb] = fminunc(f, lambda0, options)

% Build covariance matrix R in terms of length scale parameters
for i=1:nosample
    for j=1:nosample
        R(i,j) = exp(-(lambda(1)*(trpt(i,1)-trpt(j,1))^2+...
            lambda(2)*(trpt(i,2)-trpt(j,2))^2));
    end
end
residual = g-F*beta;
sigma = 1/nosample*transpose(residual)*inv(R)*residual;

% Monte carlo simulation to test the model
N=100;
for i=1:N
    E_MC = sum((KLexpansion(30000,3000)-30000)/3000);
    P_MC= randn(1);
    h = [1 P_MC E_MC]';
    r=zeros(nosample,1);
    for j=1:nosample
        r(j) = exp(-lambda(1)*(P_MC-P(j))^2-lambda(2)*(E_MC-Esum(j)))*sigma;
    end
    mu(i) = h'*beta + r'*inv(R)*residual;
end

        
