function lambda = ccl_learna_retlambda(q,L,varargin)
% lambda = ccl_learna_retlambda(q,L)
% Return a function handle for lambda
% Input:
%   q                 Joint state
%   L                 Link length
%   varargin
% Output:
%   lambda            function handle for lambda
if nargin == 2
    a = 2;
    forward_xy = ccl_rob_forward(q,L);
    lambda = [2*a*forward_xy(1),-1];
else
    forward_xy = ccl_rob_forward(q,L);
    lambda = [2*randi([-10,10])*forward_xy(1),-1];
end
end
