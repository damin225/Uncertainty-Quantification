%% Generating observations

g = @(P,E1,E2,E3,E4) P.*(1./E1+1./E2+1./E3+1./E4);
Ypred = g(2000,28000,30000,32000,34000);
Nobs = 5;
for i=1:Nobs
    Yobs(i) = Ypred + randn(1)*0.01;
end

%% Bayesian updating

Nosample = 20000;
store = zeros(Nosample,5);
E1=28000; E2=30000; E3=32000; E4=34000; sigma=0.1 ;
M =500;

for i=1:Nosample
    E1_old = E1;
    E2_old = E2;
    E3_old = E3;
    E4_old = E4;
    sigma_old = sigma;
    
    E1 = normrnd(E1,3000);
    E2 = normrnd(E2,3000);
    E3 = normrnd(E3,3000);
    E4 = normrnd(E4,3000);
    sigma = normrnd(sigma,0.01);
    
    iflag = 0;
    
    % liklihood function
    Ypred = g(2000,E1,E2,E3,E4);
    
    f = 1;
    for j=1:Nobs
        f = f*(normpdf(Yobs(j),Ypred,sigma));
    end

    % compute ratio and decide to accept or reject, normal distrubution is
    % used as prior
    alpha = f*normpdf(E1,28000,3000)*normpdf(E2,30000,3000)*...
        normpdf(E3,32000,3000)*normpdf(E4,34000,3000)*...
        normpdf(sigma,0.1,0.01)/(M*(normpdf(E1,E1_old,3000)...
        *normpdf(E2,E2_old,3000)*normpdf(E3,E3_old,3000)...
        *normpdf(E4,E4_old,3000)*normpdf(sigma,sigma_old,0.01)));
    if (isnan(alpha))
        alpha = 0;
    elseif(alpha>=1)
        alpha = 1;
    end
    % reject or accept
    u = rand(1);
    if (u>alpha)
        E1 = E1_old;
        E2 = E2_old;
        E3 = E3_old;
        E4 = E4_old;
        sigma = sigma_old;
        iflag = 1;
    end
    % store data if accepted
    if iflag ~= 1
        store(i,1) = E1;
        store(i,2) = E2;
        store(i,3) = E3;
        store(i,4) = E4;
        store(i,5) = sigma;
    end
end

figure(1)
E1 = nonzeros(store(5000:end,1));
histogram(E1,50)
figure(2)
E2 = nonzeros(store(5000:end,2));
histogram(E2,50)
figure(3)
E3 = nonzeros(store(5000:end,3));
histogram(E3,50)
figure(4)
E4 = nonzeros(store(5000:end,4));
histogram(E4,50)
figure(5)
sigma = nonzeros(store(5000:end,5));
histogram(sigma,50)
h = histfit(E1)