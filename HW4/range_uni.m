function y = range_uni(E,low,upp)
if (E>=low && E<=upp)
    y = 1/(upp-low);
else
    y = 0;
end