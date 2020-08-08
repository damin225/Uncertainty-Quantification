%% Generating observations

g = @(P,E1,E2,E3,E4) P.*(1./E1+1./E2+1./E3+1./E4);
Ypred = g(2000,10000,20000,30000,40000);
Nobs = 5;
for i=1:Nobs
    Yobs(i) = Ypred + randn(1)*0.01;
end

%% Bayesian updating

% initial guess for the parameters
Nosample = 20000;
store = zeros(Nosample,5);
E1=10000; E2=20000; E3=30000; E4=40000; sigma=0.1 ;
M=1e9;

for i=1:Nosample
    iflag = zeros(5,1);
% initialize values at the beignning of each iteration
    E1_old = E1;
    E2_old = E2;
    E3_old = E3;
    E4_old = E4;
    sigma_old = sigma;
    
    E1 = randn(1)*1000+E1;
    E2 = randn(1)*2000+E2;
    E3 = randn(1)*3000+E3;
    E4 = randn(1)*4000+E4;
    sigma = randn(1)*0.01+sigma;

% check E1
    % liklihood function
    Ypred = g(2000,E1,E2_old,E3_old,E4_old);
    f = 1;
    for j=1:Nobs
        f = f*normpdf(Yobs(j),Ypred,sigma_old);
    end
    % compute ratio and decide to accept or reject
    alpha = f/(M*(normpdf(E1,E1_old,1000)...
        *normpdf(E2_old,E2_old,2000)*normpdf(E3_old,E3_old,3000)...
        *normpdf(E4_old,E4_old,4000)*normpdf(sigma_old,sigma_old,0.01)));

    u = rand(1);
    if (u<=alpha)
        E1 = E1;
    else
        E1 = E1_old;
        iflag(1) = 1;
    end
    
% check E2
    % liklihood function
    Ypred = g(2000,E1,E2,E3_old,E4_old);
    f = 1;
    for j=1:Nobs
        f = f*normpdf(Yobs(j),Ypred,sigma_old);
    end
    % compute ratio and decide to accept or reject
    alpha = f/(M*(normpdf(E1,E1_old,1000)...
        *normpdf(E2,E2_old,2000)*normpdf(E3_old,E3_old,3000)...
        *normpdf(E4_old,E4_old,4000)*normpdf(sigma_old,sigma_old,0.01)));

    u = rand(1);
    if (u<=alpha)
        E2 = E2;
    else
        E2 = E2_old;
        iflag(2) = 1;
    end
    
% check E3
    % liklihood function
    Ypred = g(2000,E1,E2,E3,E4_old);
    f = 1;
    for j=1:Nobs
        f = f*normpdf(Yobs(j),Ypred,sigma_old);
    end
    % compute ratio and decide to accept or reject
    alpha = f/(M*(normpdf(E1,E1_old,1000)...
        *normpdf(E2,E2_old,2000)*normpdf(E3,E3_old,3000)...
        *normpdf(E4_old,E4_old,4000)*normpdf(sigma_old,sigma_old,0.01)));

    u = rand(1);
    if (u<=alpha)
        E3 = E3;
    else
        E3 = E3_old;
        iflag(3) = 1;
    end
    
% check E4
    % liklihood function
    Ypred = g(2000,E1,E2,E3,E4);
    f = 1;
    for j=1:Nobs
        f = f*normpdf(Yobs(j),Ypred,sigma_old);
    end
    % compute ratio and decide to accept or reject
    alpha = f/(M*(normpdf(E1,E1_old,1000)...
        *normpdf(E2,E2_old,2000)*normpdf(E3,E3_old,3000)...
        *normpdf(E4,E4_old,4000)*normpdf(sigma_old,sigma_old,0.01)));

    u = rand(1);
    if (u<=alpha)
        E4 = E4;
    else
        E4 = E4_old;
        iflag(4) = 1;
    end

% check sigma
    % liklihood function
    Ypred = g(2000,E1,E2,E3,E4);
    f = 1;
    for j=1:Nobs
        f = f*normpdf(Yobs(j),Ypred,sigma);
    end
    % compute ratio and decide to accept or reject
    alpha = f/(M*(normpdf(E1,E1_old,1000)...
        *normpdf(E2,E2_old,2000)*normpdf(E3,E3_old,3000)...
        *normpdf(E4,E4_old,4000)*normpdf(sigma,sigma_old,0.01)));

    u = rand(1);
    if (u<=alpha)
        sigma = sigma;
    else
        sigma = sigma_old;
        iflag(5) = 1;
    end
    
    if iflag(1) ~= 1
        store(i,1) = E1;
    end
    if iflag(2) ~= 1
        store(i,2) = E2;
    end
    if iflag(3) ~= 1
        store(i,3) = E3;
    end
    if iflag(4) ~= 1
        store(i,4) = E4;
    end
    if iflag(5) ~= 1
        store(i,5) = sigma;
    end
end
E3 = nonzeros(store(18000:end,1));
histogram(E3,100)


  
    
    
    
    
    
    
    