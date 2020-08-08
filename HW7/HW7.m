% open temperature data from file
fid = fopen('temperature.txt');
Tdata = fscanf(fid,'%f');
Tori = transpose(reshape(Tdata,[81,500]));
Tuse = Tori(1:450,:);

% compute correlation matrix
Tcor = corr(Tuse);
[V,D] = eig(Tcor);
T = transpose(V);

% verify the results
for i=451:455
    W = T(1:3,:); % only use first three components
    Ttest = Tori(i,:);
    yR = W*transpose(Ttest);
    WR = V(:,1:3);
    TR = WR*yR;
    varPercent = var(TR)/var(Ttest)
end