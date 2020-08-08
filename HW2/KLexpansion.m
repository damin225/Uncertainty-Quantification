function Esample = KLexpansion(mean, std)

% Autocorrelation function
R = @(tau) exp(-abs(tau));

% Generate correlation matrix
t = [0:0.25:0.75]; 
n = length(t);
Rmatrix = zeros(n);
for i=1:n
    for j=1:n
        Rmatrix(i,j) = R((j-i)*0.25);
    end
end
Rmatrix = std^2*Rmatrix;

% Generate eigval and eigvec
[V,D] = eig(Rmatrix);
lambda = diag(D);

% Generate random process
E = zeros(4,1);
for i=4:-1:1
    zeta = randn(1);
    E = E + sqrt(lambda(i))*V(:,i)*zeta;
end

Esample = E + mean;
