function functionHandle = def_null_space_proj(constraintMatrix)
% Defines null space projection matrix. 
%
% Given a MatLab function handle to a constraint matrix A(x)
% function of the state x, def_null_space_proj returns the MatLab
% function handle to the null space projection matrix as,
%
%     N(x) = I - pinv(A) * A.
%
% Syntax: functionHandle = def_null_space_proj(constraintMatrix)
%
% Inputs:
%    constraintMatrix - MatLab function handle to constraint Matrix
%
% Outputs:
%    functionHandle - MatLab function handle with robot configuration 
%                     (column vector) as input
%

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 29-Oct-2017

%------------- BEGIN CODE --------------
functionHandle = @nullSpaceProjection;
function output = nullSpaceProjection(q)
    A = constraintMatrix(q); % Compute constraint matrix for given configuration.
    Ainv = pinv(A); % Pseudo inverse of constraint matrix.
    output = eye(length(q)) - Ainv*A; % Compute null-space projection matrix for given configuration.
end
%------------- END OF CODE --------------
end
