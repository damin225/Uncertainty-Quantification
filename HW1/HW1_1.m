% Spectral density function
s = @(w) 1/(pi*(1+w^2));

% Autocorrelation
R = @(tau) exp(-abs(tau));

% Generate random process and verify the results
N = 500; delta_w = 0.016*pi;
t = [0:0.01:5];
count = length(t);
no_sample = 750;
y = zeros(1,count);
X = zeros(no_sample,count);
for j=1:no_sample
    g=0;
    for i = 1:N
        phi = 2*pi*rand(1);
        g = g + sqrt(2)*((2*s(delta_w*(i-1))*delta_w)^0.5)*cos(delta_w*(i-1).*t+phi);
    end
    X(j,:) = g;
end

for i=1:100
    temp1 = [1:i:count-i];
    temp2 = [1+i:i:count];
    X1 = reshape([X(:,temp1)],[length(temp1)*no_sample,1]);
    X2 = reshape([X(:,temp2)],[length(temp2)*no_sample,1]);
    rho_mat = corrcoef(X1,X2);
    rho(i+1) = rho_mat(1,2);
end
rho(1) = 1;
tau = [0:0.01:1];
hold on
scatter(tau,rho,'.');
plot(tau,R(tau),'linewidth',2);
legend('simulated','Target');
ylim([0:1.2]);
set(gca,'FontSize', 24);
xlabel('\tau');
ylabel('R(\tau)');
hold off
