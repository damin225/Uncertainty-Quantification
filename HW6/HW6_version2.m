%% Build GPR model
% Generate trainning points for 2nd order surrogate model
Etrain = zeros(4,3);
for i=1:4
    Mean(i) = 24000+(i-1)*2000;
    Std(i) = Mean(i)*0.1;
    Etrain(i,:) = [Mean(i)-Std(i), Mean(i), Mean(i)+Std(i)];
end

trpt = zeros(81,4);
for i=1:4
    trpt(:,i) = repmat(transpose(repelem(Etrain(i,:),3^(4-i))),3^(i-1),1);
end

P = 2000;
g = @(P,E1,E2,E3,E4) P.*(1./E1+1./E2+1./E3+1./E4);
Y = g(P,trpt(:,1),trpt(:,2),trpt(:,3),trpt(:,4));

% Train Gaussian Process model using the trainning points
gprMdl = fitrgp(trpt,Y,'Basis','linear','FitMethod','exact',...
                'PredictMethod','exact');
            
%% PCE
X = [ones(81,1) trpt trpt(:,1).^2 trpt(:,2).^2 trpt(:,3).^2 trpt(:,4).^2 ...
    trpt(:,1).*trpt(:,2) trpt(:,1).*trpt(:,3) trpt(:,1).*trpt(:,4)...
    trpt(:,2).*trpt(:,3) trpt(:,2).*trpt(:,4) trpt(:,3).*trpt(:,4)];
b = (X'*X)\(X'*Y);
 
s = @(E1,E2,E3,E4) b(1)+b(2).*E1+b(3).*E2+b(4).*E3+b(5).*E4+...
    b(6)*E1.^2+b(7)*E2.^2+b(8)*E3.^2+b(9)*E4.^2+b(10)*E1.*E2+b(11)*E1.*E3+...
    b(12)*E1.*E4+b(13)*E2.*E3+b(14)*E2.*E4+b(15)*E3.*E4;
            
%% 1st order sobol indices
nosample = 100;
E = zeros(4,nosample);
Esample = zeros(4,nosample);

for i=1:4 
    for j=1:nosample %outter loop
        E(i,j) = randn(1)*Std(i)+Mean(i);
        for k=1:nosample %inner loop
            for l=1:4
                if (l~=i)
                    Esample(l,k) = randn(1)*Std(l)+Mean(l);
                else
                    Esample(l,k) = E(i,j);
                end 
            end
            y(k) = predict(gprMdl,[Esample(1,k), Esample(2,k),...
                           Esample(3,k),Esample(4,k)]);
            yPCE(k) = s(Esample(1,k), Esample(2,k),...
                           Esample(3,k),Esample(4,k));
            yPhy(k) = g(P,Esample(1,k), Esample(2,k),...
                           Esample(3,k),Esample(4,k));
        end
        ymean(j) = mean(y);
        ymeanPCE(j) = mean(yPCE(k));
        ymeanPhy(j) = mean(yPhy(k));
    end
    Vi(i) = var(ymean);
    ViPCE(i) = var(ymeanPCE);
    ViPhy(i) = var(yPhy);
end
Si = Vi/sum(Vi);
SiPCE = ViPCE/sum(ViPCE);
SiPhy = ViPhy/sum(ViPhy);

%% total effect index
for i=1:4 
    for j=1:nosample %outter loop
            for l=1:4
                if (l~=i)
                    Esample(l,j) = randn(1)*Std(l)+Mean(l);
                end
            end
        for k=1:nosample %inner loop
            Esample(i,k) = rand(1)*Std(i)+Mean(i);
            y(k) = predict(gprMdl,[Esample(1,k), Esample(2,k),...
                           Esample(3,k),Esample(4,k)]);
            yPCE(k) = s(Esample(1,k), Esample(2,k),...
                           Esample(3,k),Esample(4,k));
            yPhy(k) = g(P,Esample(1,k), Esample(2,k),...
                           Esample(3,k),Esample(4,k));
        end
        ymean(j) = mean(y);
        ymeanPCE(j) = mean(yPCE(k));
        ymeanPhy(j) = mean(yPhy(k));
    end
    V1(i) = var(ymean);
    V1PCE(i) = var(ymeanPCE);
    V1Phy(i) = var(yPhy);
end
S1 = 1-V1/sum(V1);
S1PCE = 1-V1PCE/sum(V1PCE);
S1Phy = 1-V1Phy/sum(V1Phy);

   