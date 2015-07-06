
function [nupe, v, mse] = get_nupe(F, Fp)

[d v mse] = get_nmse(F,Fp);
nupe = sum(mse)/sum(v);

