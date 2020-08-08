%% Generating observations

g = @(P,E1,E2,E3,E4) P.*(1./E1+1./E2+1./E3+1./E4);
Ypred = g(2000,28000,30000,32000,34000);
Nobs = 5;
for i=1:Nobs
    Yobs(i) = Ypred + randn(1)*0.005;
end

%% Particle filter method
nosample = 10000;
for i=1:5
    if (i==1)
        for j=1:nosample
            % prior: uniform distribution
            E1(j) = rand(1)*8000+25000;
            E2(j) = rand(1)*8000+27000;
            E3(j) = rand(1)*8000+29000;
            E4(j) = rand(1)*8000+31000;
            sigma(j) = rand(1)*0.01;
            % compute likelihood
            Ypred = g(2000,E1(j),E2(j),E3(j),E4(j));
            L(j) = normpdf(Yobs(i),Ypred,sigma(j));
        end
        % normalize L to get weights
        L = L/sum(L);
        cdf = [0,cumsum(L)];
        
    else
        for j=1:nosample
            u = rand(1);
            for k = 1:length(cdf)
                if (u>=cdf(k) && u<cdf(k+1))
                    resample_no = k;
                end
            end
            % resampling based on weigths
            E1_new(j) = E1(resample_no);
            E2_new(j) = E2(resample_no);
            E3_new(j) = E3(resample_no);
            E4_new(j) = E4(resample_no);
            sigma_new(j) = sigma(resample_no);
            % compute likelihood
            Ypred = g(2000,E1_new(j),E2_new(j),E3_new(j),E4_new(j));
            L(j) = normpdf(Yobs(i),Ypred,sigma_new(j));
        end

        % normalize L to get weights
        L = L/sum(L);
        cdf = [0,cumsum(L)];
        
        E1 = E1_new;
        E2 = E2_new;
        E3 = E3_new;
        E4 = E4_new;
        sigma = sigma_new; 
    end
end
figure (1)
histogram(E1,10)
title('posterior of E1')

figure (2)
histogram(E2,10)
title('posterior of E2')

figure (3)
histogram(E3,10)
title('posterior of E3')

figure (4)
histogram(E4,10)
title('posterior of E4')

figure (5)
histogram(sigma,10)
title('posterior of \sigma')
