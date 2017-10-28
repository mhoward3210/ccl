function functionHandle = getTaskRegressors4SurfacePerpendicularMotionSimulated(robotHandle)
% getTaskRegressors4SurfacePerpendicularMotionSimulated - Get regressors for
% the main task when representing the task vector as a product of a vector
% of regressors by a constant matrix.
%
% Syntax:  functionHandle = GETTASKREGRESSORS4SRFACEPERTENDCULARMOTIONSIMULATED(robotHandle)
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
% % Phi_b(x): vector of regressors for the main task as a function of the
% % state
% Phi_b = getTaskRegressors4SurfacePerpendicularMotionSimulated(robot);
% n = [0; 0; 1];
% W_A = blkdiag(n.', n.', n.'); % constant gain matrix for the Constraint matrix
% W_b = -5*[W_A [-n.'*[0.4; 0.0; 0.0]; 0; 0]];
% % main task as a function of the configuration
% b = @(x) W_b*Phi_b(x);
% % Constraint matrix for given robot arm configuration
% disp(b([0;0;0;pi/2;0;-pi/2;0]));
%
% Libraries required: Peter Corke's Robotics library (MatLab add-on)
% 
% See also: GETCONSTRAINEDMATRIXREGRESSORS4SREFACEPERPENDICULARMOTION

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 18-Oct-2017

%------------- BEGIN CODE --------------
    functionHandle = @Phi_b;
    function output = Phi_b(q)
        T = robotHandle.fkine(q); % end-effector homogeneous transformation
        tT = reshape(transl(TT),[],1); % end-effector position
        rot = t2r(T); % end-effector orientation (rotation matrix)
        xT = rot(:,1); yT = rot(:,2); % Unit vectors that define the plane perpendicular to end-effector
        output = [tT; xT; yT; 1];
    end
%------------- END OF CODE --------------
end
