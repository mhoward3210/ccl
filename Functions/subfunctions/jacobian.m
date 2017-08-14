function J = jacobian (q, L)
% J = jacobian (q,L)
% Jacobian calculation
% Input:
%   q               Joint state variable
%   L               Link length
% Output:
%   J               Jacobian
J(1,1) = -L(1)*sin(q(1,:)) - L(2)*sin(q(1,:)+q(2,:)) ;
J(1,2) =                   - L(2)*sin(q(1,:)+q(2,:)) ;
J(2,1) =  L(1)*cos(q(1,:)) + L(2)*cos(q(1,:)+q(2,:)) ;
J(2,2) =                     L(2)*cos(q(1,:)+q(2,:)) ;
end
