clear
clc
% Generate trainning points for 2nd order surrogate model
P = [1200 2000 2800]';
E = [24000 30000 36000]';
trpt = zeros(243,5);
trpt(:,1) = repelem(P,3^4);
for i=5:-1:2
    trpt(:,i)=repmat(repelem(E,3^(5-i)),3^(i-1),1);
end
 
g = @(P,E1,E2,E3,E4) P.*(1./E1+1./E2+1./E3+1./E4);
Y = g(trpt(:,1),trpt(:,2),trpt(:,3),trpt(:,4),trpt(:,5));
 
% Build the surrogate model
X = [ones(243,1) trpt trpt(:,1).^2 trpt(:,2).^2 trpt(:,3).^2 trpt(:,4).^2 ...
    trpt(:,5).^2 trpt(:,1).*trpt(:,2) trpt(:,1).*trpt(:,3) trpt(:,1).*trpt(:,4)...
    trpt(:,1).*trpt(:,5) trpt(:,2).*trpt(:,3) trpt(:,2).*trpt(:,4) trpt(:,2).*trpt(:,5)...
    trpt(:,3).*trpt(:,4) trpt(:,3).*trpt(:,5) trpt(:,4).*trpt(:,5)];
b = (X'*X)\(X'*Y);
 
s = @(P,E1,E2,E3,E4) b(1)+b(2).*P+b(3).*E1+b(4).*E2+b(5).*E3+b(6).*E4+...
    b(7)*P.^2+b(8)*E1.^2+b(9)*E2.^2+b(10)*E3.^2+b(11)*E4.^2+b(12)*P.*E1...
    +b(13)*P.*E2+b(14)*P.*E3+b(15)*P.*E4+b(16)*E1.*E2+b(17)*E1.*E3+...
    b(18)*E1.*E4+b(19)*E2.*E3+b(20)*E2.*E4+b(21)*E3.*E4;
 
% Train Gaussian Process model using the trainning points
gprMdl = fitrgp(trpt,Y,'Basis','linear','FitMethod','exact',...
                'PredictMethod','exact');
            
% Monte Carlo simulation 
N = 1000;
for i=1:N
    E_MC = KLexpansion(30000,3000);
    P_MC = randn(1)*400+2000;
    Ori(i) = g(P_MC, E_MC(1),E_MC(2),E_MC(3),E_MC(4));
    PCE(i) = s(P_MC, E_MC(1),E_MC(2),E_MC(3),E_MC(4));
    GPR(i) = predict(gprMdl,[P_MC, E_MC(1),E_MC(2),E_MC(3),E_MC(4)]);
end
 
 
% Plot the distribution of displacement
figure(1)
histogram(Ori,100)
figure(2)
histogram(PCE,100);
figure(3)
histogram(GPR,100);
 
% Compare the statistic of original model, surrogate model and Guassian
% Process Model
Ori_Mean = mean(Ori)
PCE_Mean = mean(PCE)
GPR_Mean = mean(GPR)
 
Ori_std = std(Ori)
PCE_std = std(PCE)
GPR_std = std(GPR)


