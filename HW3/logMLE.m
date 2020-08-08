% program to define log max likelihood function
function y = logMLE(nosample, trpt, residual, lambda)

% Build covariance matrix R in terms of length scale parameters
for i=1:nosample
    for j=1:nosample
        R(i,j) = lambda(6)^2*exp(-(lambda(1)*(trpt(i,1)-trpt(j,1))^2+...
            lambda(2)*(trpt(i,2)-trpt(j,2))^2+lambda(3)*(trpt(i,3)-trpt(j,3))^2+...
            lambda(4)*(trpt(i,4)-trpt(j,4))^2+lambda(5)*(trpt(i,5)-trpt(j,5))^2));
    end
end
%R(end,end) = R(end,end)+1e-20;
% log max likelihood function
% if (det(R)>1e-10)
%     det(R)
% end
y = -(-1/nosample*log(det(R))-log(1/nosample*(transpose(residual)*inv(R)*(residual))));