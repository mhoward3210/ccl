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
% Example: 
%     % Robot Kinematic model specified by the Denavit-Hartenberg:
%     DH = [0.0, 0.31, 0.0, pi/2;
%           0.0, 0.0, 0.0, -pi/2;
%           0.0, 0.4, 0.0, -pi/2;
%           0.0, 0.0, 0.0, pi/2;
%           0.0, 0.39, 0.0, pi/2;
%           0.0, 0.0, 0.0, -pi/2;
%           0.0, 0.21-0.132, 0.0, 0.0];
%     % Peters Cork robotics library has to be installed:
%     robot = SerialLink(DH);
%     % Phi_A(x): vector of regressors for the Constraint matrix as a function
%     % of the state
%     n = [0; 0; 1]; % Cartesian normal of the constraint surface.
%     W_A = blkdiag(n.', n.', n.'); % Constant gain matrix for the Constraint matrix.
%     Phi_A = def_phia_4_spm(robot);
%     % Constraint matrix as a function of configuration.
%     A = @(x) W_A*PhiA(x);
%     % Defining null space projection matrix
%     N = def_null_space_proj(A);
%     % Null space projection matrix for given robot arm configuration:
%     x = [0;0;0;pi/2;0;-pi/2;0];
%     disp(N(x));

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
