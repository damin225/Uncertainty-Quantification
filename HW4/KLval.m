function KL = KLval(E, mean, std)
    KL=0;
    n=length(E);
    pd = fitdist(E,'Normal');
    for i=1:n
        y(i)= normpdf(E(i),pd.mu,pd.sigma)*log((normpdf(E(i),pd.mu,pd.sigma))/normpdf(E(i),mean,std));
    end
    KL = trapz(y);