function functionHandle = def_constrained_policy(constraintMatrix, task, unconstrainedPolicy)
% Defines a constrained policy given the unconstrained policy and the constraint.
%
% Consider the decomposition of the robot actions as a main task and
% a secondary task in the null space of the main:
%
%       u(x) = pinv(A(x)) * b(x) + (I - pinv(A(x)) * A(x)) * u_pi(x),
%       
% where x is the state (robot configuration), A(x) a Pfaffian constraint matrix,
% and u_pi(x) is the unconstrained policy for the secondary task.
% def_constrained_policy returns a MatLab function handle to the unconstrained
% policy u(x), given A(x), b(x), and u_pi(x).
%
% Syntax: def_constrained_policy(constraintMatrix, task, unconstrainedPolicy)
%
% Inputs:
%    constraintMatrix - MatLab function handle to constraint Matrix;
%    task - MatLab function handle to the main task;
%    unconstrainedPolicy - MatLab handle to the unconstrainde policy.
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
%     % Phi_b(x): vector of regressors for the main task as a function of the
%     % state:
%     Phi_b = def_phib_4_spm_sim(robot);
%     % Main task as a function of the configuration:
%     b = @(x) W_b*Phi_b(x);
%     % Defining unconstrained policy regressors:
%     centre = [0.1; 0.0; 0.4];
%     radius = 0.02;
%     Phi = def_phi_4_cwm(robot, centre, radius);
%     % Defining unconstrained policy:
%     u_pi = @(x) Phi(x)*[1 10];
%     % Defining constrained policy
%     pi = def_constrained_policy(A, b, u_pi);
%     % Constrained policy for given robot arm configuration:
%     x = [0;0;0;pi/2;0;-pi/2;0];
%     disp(pi(x));

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 29-Oct-2017

%------------- BEGIN CODE --------------
functionHandle = @constrainedPolicy; % Return handle of constrained policy.
function output = constrainedPolicy(q)
    A = constraintMatrix(q); % Compute constraint matrix for given configuration.
    Ainv = pinv(A); % Pseudo inverse of constraint matrix.
    N = eye(length(q)) - Ainv*A; % Compute null-space projection matrix for given configuration.
    b = task(q); % Compute task vector for given configuration.
    pi = unconstrainedPolicy(q); % Compute unconstrained policy.
    output = Ainv*b + N*pi; % Output constrained policy.
end
%------------- END OF CODE --------------
end
