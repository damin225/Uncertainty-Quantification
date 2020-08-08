clear
clc
%% GPR model
X=[11 7 10 9 9 9 9 1  10 11; 14 21 25 15 13 22 23 30 29 13]';
Y=[343 487 705 323 272 571 629 1031 944 298]';

sigma0 = std(Y);
sigmaF0 = sigma0;
d = size(X,2);
sigmaM0 = 10*ones(d,1);
gprMdl = fitrgp(X,Y,'Basis','linear','FitMethod','exact',...
                'PredictMethod','exact','KernelFunction','ardsquaredexponential','KernelParameters',[sigmaM0;sigmaF0]);
            
%% 1st order sobol indices
% nosample = 100;
% Std = [2,5];
% Mean = [10,20];
% for i=1:2 
%     for j=1:nosample %outter loop
%         Xsample(i,j) = randn(1)*Std(i)+Mean(i);
%         for k=1:nosample %inner loop
%             for l=1:2
%                 if (l~=i)
%                     Xsample(l,k) = randn(1)*Std(l)+Mean(l);
%                 else
%                     Xsample(l,k) = Xsample(i,j);
%                 end 
%             end
%             y(k) = predict(gprMdl,[Xsample(1,k), Xsample(2,k)]);
%         end
%         ymean(j) = mean(y);
%     end
%     Vi(i) = var(ymean);
% end
% Si = Vi/sum(Vi);

%% MCMC metho to estimate the paramter of model discrepency
% Xobs = [7 9 11;21 15 13]';
% Yobs = [510 340 280]';
% nosample=1000;
% 
% for i=1:3
%     Ym = predict(gprMdl,[Xobs(i,1), Xobs(i,2)]);
%     delta_obs(i) = Ym-Yobs(i);
% end
% 
% b0=1; b1=1; lambda1=1; lmabda2=1; sigma=1;
% for i=1:Nosample
%     b0_old = b0;
%     b1_old = b1;
%     lambda1_old = lambda1;
%     lambda2_old = lambda2;
%     sigma_old = sigma;
%     
%     b0 = normrnd(b0,0.02);
%     b1 = normrnd(b1,0.02);
%     lambda1 = normrnd(lambda1,0.01);
%     lambda2 = normrnd(lambda2,0.01);
%     sigma = normrnd(sigma,0.01);
%     
%     iflag = 0;
%     
%     % liklihood function
%     Ypred = g(2000,b0,b1,lambda1,lambda2);
%     
%     f = 1;
%     for j=1:Nobs
%         f = f*(normpdf(Yobs(j),Ypred,sigma));
%     end
% 
%     % compute ratio and decide to accept or reject, normal distrubution is
%     % used as prior
%     alpha = f*normpdf(b0,28000,3000)*normpdf(b1,30000,3000)*...
%         normpdf(lambda1,32000,3000)*normpdf(lambda2,34000,3000)*...
%         normpdf(sigma,0.1,0.01)/(M*(normpdf(b0,b0_old,3000)...
%         *normpdf(b1,b1_old,3000)*normpdf(lambda1,lambda1_old,3000)...
%         *normpdf(lambda2,lambda2_old,3000)*normpdf(sigma,sigma_old,0.01)));
%     if (isnan(alpha))
%         alpha = 0;
%     elseif(alpha>=1)
%         alpha = 1;
%     end
%     % reject or accept
%     u = rand(1);
%     if (u>alpha)
%         b0 = b0_old;
%         b1 = b1_old;
%         lambda1 = lambda1_old;
%         lambda2 = lambda2_old;
%         sigma = sigma_old;
%         iflag = 1;
%     end
%     % store data if accepted
%     if iflag ~= 1
%         store(i,1) = b0;
%         store(i,2) = b1;
%         store(i,3) = lambda1;
%         store(i,4) = lambda2;
%         store(i,5) = sigma;
%     end
% end

%% Bayes factor
Xobs = [7 9 11;21 15 13]';
Yobs = [510 340 280]';
for i=1:3
    Y = predict(gprMdl,[Xobs(i,1),Xobs(i,2)]);
    L(i) = normpdf(Yobs(i),Y,10);
    
    X1 = rand(1)*2+10;
    X2 = rand(1)*5+20;
    Y1 = predict(gprMdl,[X1,X2]);
    L1(i) = normpdf(Yobs(i),Y1,10);
    
    B(i) = L(i)/L1(i);
end