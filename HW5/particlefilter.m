%UQ
%HW# 5
%Ollie Stover
%particle filter


clear
clc

L = 1; %length of bar, convert from ft to inches
A = 1; % area of bar, convert from ft^2 to inches^2;
t=4; %number of discretizations in bar

N = 5000;% number of particles


 %define statistics of E
 COV = .1;
muE1 = 30000;
sigmaE1 = 3000;
muE2 = 40000;
sigmaE2 = COV*muE2;
muE3 = 20000;
sigmaE3 = COV*muE3;
muE4 = 10000;
sigmaE4 = COV*muE4;



%generate true values of E
E1t =normrnd(muE1, sigmaE1);
E2t =normrnd(muE2, sigmaE2);
E3t =normrnd(muE3, sigmaE3);
E4t =normrnd(muE4, sigmaE4);
Ytrue = predict([E1t, E2t, E3t, E4t]); %calculate true displacement
Yobs = zeros(1,5);

%generate 5 observations = predicted value+random value
for r = 1:5
    Yobs(r) = Ytrue+normrnd(0, .01);

end
%predefine matrices to use later
E = zeros(N, 4);
Particles = zeros(N,5,6);

%assume uniform priors - define prior regions
rangeE1 = [muE1-2*sigmaE1, muE1+2*sigmaE1];
rangeE2 = [muE2-2*sigmaE2, muE2+2*sigmaE2];
rangeE3 = [muE3-2*sigmaE3, muE3+2*sigmaE3];
rangeE4 = [muE4-2*sigmaE4, muE4+2*sigmaE4];
rangesigma = [0, .2];

%generate 5000 inital particles

out = @(range) range(1) + (range(2)-range(1))*rand(N,1);
Particles(:,1,1) = out( rangeE1);
Particles(:,2,1) = out( rangeE2);
Particles(:,3,1) = out(rangeE3);
Particles(:,4,1) = out( rangeE4);
Particles(:,5,1) = out(rangesigma);

%initialize matrices to use later
L = zeros(N,5); %liklihood matrix
w = zeros(N,5); %weight function
cdf = w; %initialize cdf
Ypred= zeros(N,5); %predicted value


for i = 1:5 %loop over observed values
    for j = 1:N %loop over numbr of particles
    Ypred(j,i) = predict(Particles(j,1:4,i)); %calculated predicted displacement value
    L(j,i) = normpdf((Yobs(i)-Ypred(j,i))/Particles(j,5,i)); %calc likilihood value
    end
    temp = sum(L(:,i));
    w(:,i) = L(:,i)./temp; %normalize the weights
    %sort weights and particles by order of weights
    [w(:,i), index] = sortrows(w(:,i));
    Particles(:,:, i) = Particles(index, :, i);
    %develop CDF
    cdf(:,i) = cumsum(w(:,i));
    %resample 5000 new particles
    for j = 1:N
    test = rand(1,N);
    for k = 1:N-1
        if cdf(k,i) < test(j) && cdf(k+1,i)>test(j)
        Particles(j,:,i+1) = Particles(k, :, i);
        end
    end
    end
 
    
end

%plot results
% figure(1)
% histogram(Particles(:,1,1))
% title("Prior of E1")
% 
% figure(2)
% histogram(Particles(:,2,1))
% title("Prior of E2")
% 
% figure(3)
% histogram(Particles(:,3,1))
% title("Prior of E3")
% 
% figure(4)
% histogram(Particles(:,3,1))
% title("Prior of E4")
% 
% figure(5)
% histogram(Particles(:,3,1))
% title("Prior of sigma")

figure(6)
histogram(Particles(:,1,6))
title("Posterior of E1")

figure(7)
histogram(Particles(:,2,6))
title("Posterior of E2")

figure(8)
histogram(Particles(:,3,5))
title("Posterior of E3")

figure(9)
histogram(Particles(:,4,6))
title("Posterior of E4")

figure(10)
histogram(Particles(:,5,6))
title("Posterior of Sigma")
% 
% temp = Particles(:,1,6)+Particles(:,2,6)+Particles(:,3,6)+Particles(:,4,6);
% figure(11)
% histogram(temp)





