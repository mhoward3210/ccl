function lambda = return_lambda(q,L,varargin)
% lambda = return_lambda(q,L)
% Return a function handle for lambda
% Input:
%   q                 Joint state
%   L                 Link length
%   varargin
% Output:
%   lambda            function handle for lambda
if nargin == 2
    a = 2;
    forward_xy = forward(q,L);
    lambda = [2*a*forward_xy(1),-1];
else
    forward_xy = forward(q,L);
    lambda = [2*randi([-10,10])*forward_xy(1),-1];
end
end
