
function [nmse v mse] = get_nmse(Y,Yp)

N    = size(Y,2);          % get no. data points
mse  = sum((Y-Yp).^2,2)/N; % get mean squared error
v    = var(Y,0,2);         % get variance
nmse = mse/v;              % compute nmse

