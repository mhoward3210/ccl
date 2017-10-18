function functionHandle = getConstraintMatrixRegressor4SurfacePerpendicularMotion(robotHandle)
%GETCONSTRAINEDMATRIXREGRESSORS4SREFACEPERPENDICULARMOTION - Get regressors 
% for the constraint matrix when representing the constraint matrix as the 
% product of a vector of regressors by a constant matrix.
%
% Syntax:  functionHandle = getConstraintMatrixRegressor4SurfacePerpendicularMotion(robotHandle)
%
% Inputs:
%    robotHandle - Peter Corke's Serial-link robot class
%
% Outputs:
%    functionHandle - function to be evaluated 
%
% Example: 
% % Robot Kinematic model specified by the Denavit-Hartenberg
% DH = [0.0, 0.31, 0.0, pi/2;
%       0.0, 0.0, 0.0, -pi/2;
%       0.0, 0.4, 0.0, -pi/2;
%       0.0, 0.0, 0.0, pi/2;
%       0.0, 0.39, 0.0, pi/2;
%       0.0, 0.0, 0.0, -pi/2;
%       0.0, 0.21-0.132, 0.0, 0.0];
% % Peters Cork robotics library has to be installed
% robot = SerialLink(DH);
% % Phi_A(x): vector of regressors for the Constraint matrix as a function
% % of the state
% n = [0; 0; 1];
% W_A = blkdiag(n.', n.', n.'); % constant gain matrix for the Constraint matrix
% Phi_A = getConstraintMatrixRegressor4SurfacePerpendicularMotion(robot);
% % Constraint matrix as a function of configuration
% A = @(x) W_A*Phi_A(x);
% % Constraint matrix for given robot arm configuration
% disp(A([0;0;0;pi/2;0;-pi/2;0]));
%
% Libraries required: Peter Corke's Robotics library (MatLab add-on)
% 
% See also: GETTASKREGRESSORS4SRFACEPERTENDCULARMOTIONSIMULATED

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 18-Oct-2017

%------------- BEGIN CODE --------------
functionHandle = @Phi_A;
function output = Phi_A(q)
    J = robotHandle.jacob0(q); % Robot Jacobian in the global reference frame
    JtT = J(1:3,:); % Jacobian for the end-effector position
    Jrot = J(4:6,:); % rotation component of Jacobian
    rot = t2r(robotHandle.fkine(q)); % end-effector orientation (rotation matrix)
    xT = rot(:,1); yT = rot(:,2); % Unit vectors that define the plane perpendicular to end-effector
    JxT = -skew(xT)*Jrot; JyT = -skew(yT)*Jrot; % Jacobians for the end-effector frame unit vectors
    output = [JtT; JxT; JyT];
end
%------------- END OF CODE --------------
end