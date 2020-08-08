% Autocorrelation function
R = @(tau) exp(-abs(tau));

% Generate correlation matrix
t = [0:0.01:5]; 
n = length(t);
Rmatrix = zeros(n);
for i=1:n
    for j=1:n
        Rmatrix(i,j) = R((j-i)*0.01);
    end
end

% Generate eigval and eigvec
[V,D] = eig(Rmatrix);
lambda = diag(D);

% Generate random process and verify the results
N = 501;
count = length(t);
no_sample = 500;
X = zeros(no_sample,count);
for j=1:no_sample
    g=0;
    for i=N:-1:N-100
        zeta = randn(1);
        g = g + sqrt(lambda(i))*V(:,i)*zeta;
    end
    g=g';
    X(j,:) = g;
end

for i=1:100
    temp1 = [1:i:count-i];
    temp2 = [1+i:i:count];
    X1 = reshape([X(:,temp1)],[length(temp1)*no_sample,1]);
    X2 = reshape([X(:,temp2)],[length(temp2)*no_sample,1]);
    rho_mat = corrcoef(X1,X2);
    rho(i+1) = (rho_mat(1,2));
end
rho(1) = 1;
tau = [0:0.01:1];
hold on
scatter(tau,rho,'.');
plot(tau,R(tau),'linewidth',2);
legend('simulated','Target');
ylim([0:1.2]);
set(gca,'FontSize', 24);
hold off
