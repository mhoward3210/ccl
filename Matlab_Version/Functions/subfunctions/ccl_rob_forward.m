function r = ccl_rob_forward (q, L)
% r = ccl_rob_forward (q,L)
% Forward kinematic simulation
% Input:
%   q               Joint state variable
%   L               Link length
% Output:
%   r               Task space variable
r    = zeros(2,1) ;
r(1) = L(1)*cos(q(1,:)) + L(2)*cos(q(1,:)+q(2,:)) ;
r(2) = L(1)*sin(q(1,:)) + L(2)*sin(q(1,:)+q(2,:)) ;
end
