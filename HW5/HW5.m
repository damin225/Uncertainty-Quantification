%% Generating observations

g = @(P,E1,E2,E3,E4) P.*(1./E1+1./E2+1./E3+1./E4);
Ypred = g(2000,28000,30000,32000,34000);
Nobs = 5;
for i=1:Nobs
    Yobs(i) = Ypred + randn(1)*0.01;
end

%% Particle filter method
nosample = 5000;
for i=1:5
    if (i==1)
        for j=1:nosample
            % prior: uniform distribution
            E1_unsort(j) = rand(1)*40000+10000;
            E2_unsort(j) = rand(1)*40000+10000;
            E3_unsort(j) = rand(1)*40000+10000;
            E4_unsort(j) = rand(1)*40000+10000;
            sigma_unsort(j) = rand(1)*0.2;

            % compute likelihood
            Ypred = g(2000,E1_unsort(j),E2_unsort(j),E3_unsort(j),E4_unsort(j));
            L(j) = normpdf(Yobs(i),Ypred,sigma_unsort(j));
        end
       
        % only keep unique values of L and parameters (in a sorted fashion)
        [L_uni,ia,ic] = unique(L); 
        E1 = E1_unsort(ia);
        E2 = E2_unsort(ia);
        E3 = E3_unsort(ia);
        E4 = E4_unsort(ia);
        sigma = sigma_unsort(ia);
        
        % normalize L to get weights
        L_uni = L_uni/sum(L_uni);
        [f,x] = ecdf(L_uni);
        
    else
        for j=1:nosample
            u = rand(1);
            for k = 1:length(f)
                if (u>f(k) && u<=f(k+1))
                    resample_no(j) = k+1;
                end
            end
            % resampling based on weigths
            E1_unsort(j) = E1(resample_no(j)-1);
            E2_unsort(j) = E2(resample_no(j)-1);
            E3_unsort(j) = E3(resample_no(j)-1);
            E4_unsort(j) = E4(resample_no(j)-1);
            sigma_unsort(j) = sigma(resample_no(j)-1);
            % compute likelihood
            Ypred = g(2000,E1_unsort(j),E2_unsort(j),E3_unsort(j),E4_unsort(j));
            L(j) = normpdf(Yobs(i),Ypred,sigma_unsort(j));
        end
         % only keep unique values of L and parameters
        [L_uni,ia,ic] = unique(L); 
        E1 = E1_unsort(ia);
        E2 = E2_unsort(ia);
        E3 = E3_unsort(ia);
        E4 = E4_unsort(ia);
        sigma = sigma_unsort(ia);
        
        % normalize L to get weights
        L_uni = L_uni/sum(L_uni);
        [f,x] = ecdf(L_uni);
    end
end
histogram(E1+E2+E3+E4,50)
        
        
        
        
        
        
        
        
        