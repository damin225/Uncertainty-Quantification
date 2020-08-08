function Ypred = predict(E)
t = length(E);
P = 2000;
temp1 = 0;
for i = 1:t
        temp1 = temp1 + P/(E(i));
    end
    Ypred = temp1;