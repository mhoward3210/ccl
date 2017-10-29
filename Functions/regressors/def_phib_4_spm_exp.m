function functionHandle = def_phib_4_spm_exp(robotHandle)
% Defines a set of regressors for the task of maintaining the constraint of a surface perpendicular motion.
%
% Consider a set of Pfaffian constraints modelled as:
%
%         A(x) * u(x) = b(x),
%
% where A(x) is the constraint matrix, x is the state, u(x) are the constrained actions, and
% b(x) is a task policy that ensures the system satisfies the constraint.
% Consider we model b(x) as a linear combination of a set of regressors that depend on the state x:
%
%         b(x) = W_b * Phi_b(x),
%
% where W_b is a matrix of weights, and Phi_b(x) is a matrix of regressores.
% def_phib_4_spm_exp returns a MatLab function handle to a set of regressors
% suitable for the task of maintaining the robot end-effector in contact 
% and perpendicular to a surface, for the setting of a real robot.
% This regressors are a function of the robot configuration - column vector.
%
% Syntax:  functionHandle = def_phib_4_spm_exp(robotHandle)
%
% Inputs:
%    robotHandle - Peter Corke's Serial-link robot class
%
% Outputs:
%    functionHandle - function to be evaluated 
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
%     % Peters Cork robotics library has to be installed
%     robot = SerialLink(DH);
%     % Phi_b(x): vector of regressors for the main task as a function of the
%     % state:
%     Phi_b = def_phib_4_spm_exp(robot);
%     n = [0; 0; 1];
%     W_A = blkdiag(n.', n.', n.'); % Constant gain matrix for the Constraint matrix.
%     W_b = -5*[W_A [-n.'*[0.4; 0.0; 0.0]; 0; 0]];
%     % Main task as a function of the configuration:
%     b = @(x) W_b*Phi_b(x);
%     % Constraint matrix for given robot arm configuration
%     disp(b([0;0;0;pi/2;0;-pi/2;0]));
%
% Libraries required: Peter Corke's Robotics library (MatLab add-on)
% 
% See also: def_phib_4_spm_sim, def_phia_4_spm, def_u_pi_4_cwm

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 28-Oct-2017

%------------- BEGIN CODE --------------
    functionHandle = @Phi_b;
    function output = Phi_b(q)
        T = robotHandle.fkine(q); % End-effector homogeneous transformation.
        tT = reshape(transl(T),[],1); % End-effector position.
        rot = t2r(T); % End-effector orientation (rotation matrix).
        xT = rot(:,1); yT = rot(:,2); % Unit vectors that define the plane perpendicular to end-effector.
        q_M = q*q.'; % Matrix with the second order binomials of the configuration.
        mask = triu(true(length(q))); % Mask to choose half of the elements.
        q_2order = q_M(mask); % Second order binomials of the configuration.
        output = [q_2order; q; tT; xT; yT; 1];
    end
%------------- END OF CODE --------------
end
