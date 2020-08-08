%% physical model
g = @(P,E1,E2,E3,E4) P.*(1./E1+1./E2+1./E3+1./E4);
P = 2000;

%% 1st order sobol indices
E = zeros(4,11);
Esample = zeros(4,11);
for i=1:4
    Mean(i) = 28000+(i-1)*2000;
    Std(i) = Mean(i)*0.1;
end

nosample = 50;
for i=1:4 
    E(i,:) = KLexpansion(Mean(i),Std(i));
    for j=1:nosample %outter loop
        for k=1:nos %inner loop
            for l=1:4
                if (l~=i)
                    Esample(l,k) = randn(1)*Std(l)+Mean(l);
                else
                    Esample(l,k) = E(i,j);
                end 
            end
            y(k) = g(P,Esample(1,k),Esample(2,k),Esample(3,k),Esample(4,k));
        end
        ymean(j) = mean(y);
    end
    Vi(i) = var(ymean);
end
Si = Vi/sum(Vi);

%% total effect index
for i=1:4 
    for l=1:4
        if (l~=i)
            Esample(l,:) = KLexpansion(Mean(l),Std(l));
        end
    end
    for j=1:11 %outter loop
        for k=1:11 %inner loop
            Esample(i,:) = KLexpansion(Mean(i),Std(i));
            y(k) = g(P,Esample(1,j),Esample(2,j),Esample(3,j),Esample(4,j));
        end
        ymean(j) = mean(y);
    end
    V1(i) = var(ymean);
end
S1 = 1-V1/sum(V1);

   