% Linear basis function
%
% Implements linear basis function:
%
%  phi(x) = [ x; 1 ]
%
% in: 
%     x   - input
%
% out: 
%     phi - basis function prediction
%
function phi = fn_basis_linear ( x );

phi = [x; ones(1,size(x,2))];

