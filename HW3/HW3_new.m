%% Generating trainning points

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

% surrogate model for PCE
X = [ones(nosample,1) trpt trpt(:,1).^2 trpt(:,2).^2 trpt(:,3).^2 trpt(:,4).^2 ...
    trpt(:,5).^2 trpt(:,1).*trpt(:,2) trpt(:,1).*trpt(:,3) trpt(:,1).*trpt(:,4)...
    trpt(:,1).*trpt(:,5) trpt(:,2).*trpt(:,3) trpt(:,2).*trpt(:,4) trpt(:,2).*trpt(:,5)...
    trpt(:,3).*trpt(:,4) trpt(:,3).*trpt(:,5) trpt(:,4).*trpt(:,5)];
b = (X'*X)\(X'*g);

s = @(P,E1,E2,E3,E4) b(1)+b(2).*P+b(3).*E1+b(4).*E2+b(5).*E3+b(6).*E4+...
    b(7)*P.^2+b(8)*E1.^2+b(9)*E2.^2+b(10)*E3.^2+b(11)*E4.^2+b(12)*P.*E1...
    +b(13)*P.*E2+b(14)*P.*E3+b(15)*P.*E4+b(16)*E1.*E2+b(17)*E1.*E3+...
    b(18)*E1.*E4+b(19)*E2.*E3+b(20)*E2.*E4+b(21)*E3.*E4;

%% Build Gassuian Process Model

% Data normalizing
P = (P-2000)/400;
for i=1:4
    E(:,i) = (E(:,i)-30000)/3000;
end
trpt = [P,E];

% Linear regression to get trend coefficients beta
F = [ones(nosample,1) trpt(:,1) trpt(:,2) trpt(:,3) trpt(:,4) trpt(:,5)];
beta = (F'*F)\(F'*g);

% Maximize log ML function to get parameters
residual = g-F*beta;
options = optimset('Display','off','MaxIter',2000);%,'Algorithm',...
%     'quasi-newton','HessUpdate','steepdesc','OptimalityTolerance',...
%     1e-8,'StepTolerance',1e-8,'MaxFunctionEvaluations',2000);
lambda0=ones(6,1)*0.1;
f = @(lambda) logMLE(nosample,trpt, residual, lambda);
[lambda,logProb] = fminsearch(f,lambda0, options)

% Build covariance matrix R in terms of length scale parameters
for i=1:nosample
    for j=1:nosample
        R(i,j) = lambda(6)^2*exp(-(lambda(1)*(trpt(i,1)-trpt(j,1))^2+...
            lambda(2)*(trpt(i,2)-trpt(j,2))^2+lambda(3)*(trpt(i,3)-trpt(j,3))^2+...
            lambda(4)*(trpt(i,4)-trpt(j,4))^2+lambda(5)*(trpt(i,5)-trpt(j,5))^2));
    end
end

%% Monte Carlo simulation

N = 1000;
for i=1:N
    E_MC = KLexpansion(30000,3000);
    P_MC = randn(1)*400+2000;
    Ori(i) = y(P_MC, E_MC(1),E_MC(2),E_MC(3),E_MC(4));
    PCE(i) = s(P_MC, E_MC(1),E_MC(2),E_MC(3),E_MC(4));
    
    % data normalizing
    P_MC = (P_MC-2000)/400;
    for j=1:4
        E_MC(j) = (E_MC(j)-30000)/3000;
    end
    
    h = [1; P_MC; E_MC];
    for j=1:nosample
        r(j) = lambda(6)^2*exp(-(lambda(1)*(P_MC-trpt(j,1))^2+...
            lambda(2)*(E_MC(1)-trpt(j,2))^2+lambda(3)*(E_MC(2)-trpt(j,3))^2+...
            lambda(4)*(E_MC(3)-trpt(j,4))^2+lambda(5)*(E_MC(4)-trpt(j,5))^2));
    end
    mu(i) = h'*beta + r*inv(R)*residual;
    temp = [zeros(6) F';F R];
    var(i) = lambda(6)^2-[h' r]*inv(temp)*[h;r'];
end
figure (1)
histogram(Ori,100)

figure (2)
histogram(PCE,100)

figure (3)
histogram(mu,100)




