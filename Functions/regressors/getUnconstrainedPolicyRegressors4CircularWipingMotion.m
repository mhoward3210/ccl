function functionHandle = getUnconstrainedPolicyRegressors4CircularWipingMotion(robotHandle, c_G, radius)
%  Returns a function handle for the basis functions of the wiping unconstrained policy linear basis function model.
%  The output function handle is a function of the robot arm configuration - column vector with appropriate dimension.
%
% Syntax:  functionHandle = getUnconstrainedPolicyRegressors4CircularWipingMotion(robotHandle, c_G, radius)
%
% Inputs:
% robotHandle - Peter Corke's Serial-link robot class
%    c_G - 3 dimensional column vector with Cartesian coordinates of the centre of the wiping motion relative to the robot global frame and in meters;
%    radius - radius of the wiping circle in meters
%
% Outputs:
%    functionHandle - function to be evaluated 
%
% Example: 
%     % Robot Kinematic model specified by the Denavit-Hartenberg
%     DH = [0.0, 0.31, 0.0, pi/2;
%           0.0, 0.0, 0.0, -pi/2;
%           0.0, 0.4, 0.0, -pi/2;
%           0.0, 0.0, 0.0, pi/2;
%           0.0, 0.39, 0.0, pi/2;
%           0.0, 0.0, 0.0, -pi/2;
%           0.0, 0.21-0.132, 0.0, 0.0];
%     % Peters Cork robotics library has to be installed
%     robot = SerialLink(DH);
%     % Defining unconstrained policy regressors
%     centre = [0.1; 0.0; 0.4];
%     radius = 0.02;
%     Phi = getUnconstrainedPolicyRegressors4CircularWipingMotion(robot, centre, radius);
%     % Defining unconstrained policy
%     pi_u = @(x) Phi(x)*[1 10];
%     % Constraint matrix for given robot arm configuration
%     x = [0;0;0;pi/2;0;-pi/2;0];
%     disp(A(x));
%
% Libraries required: Peter Corke's Robotics library (MatLab add-on)
% 
% See also: def_phia_4_spm

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 25-Oct-2017

%------------- BEGIN CODE --------------
functionHandle = @Phi;
function output = Phi(q)
    J = robotHandle.jacobe(q); % Robot Jacobian in the end-effector frame
    Jtask = J(1:2,:); % Jacobian for the x and y coordinates - perpendicular plane to the end-effector
    %Phi_kappa = getPhi_kappa(robotHandle, c_G, radius); % regressors for the secondary task
    %output = pinv(Jtask)*Phi_kappa(q);
    N = eye(length(q)) - (Jtask\Jtask); % Compute null-space projection matrix for given configuration
    Phi_kappa = getPhi_kappa(robotHandle, c_G, radius); % regressors for the secondary task
    Phi_gamma = @(q) kron([q.' 1],eye(length(q))); % regressors for the third task
    output = [Jtask\Phi_kappa(q) N*Phi_gamma(q)];
end
function functionHandle = getPhi_kappa(robotHandle, c_G, radius)
    functionHandle = @Phi_kappa;
    function output = Phi_kappa(q)
        c_ro = feval(getC_ro(robotHandle, c_G),q); % centre of the circular motion to the end-effector relative to the end-effector frame
        c_ro_per = [0 -1; 1 0]*c_ro; % perpendicular to c_ro
        nc_ro = norm(c_ro); % total distance to the centre
        output = [c_ro_per c_ro*(1-(radius/nc_ro))];
    end
    function functionHandle = getC_ro(robotHandle, c_G)
        functionHandle = @c_ro;
        function output = c_ro(q)
            T = robotHandle.fkine(q); % end-effector homogeneous transformation
            tT = transl(T).'; % end-effector position
            R = t2r(T); % end-effector orientation (rotation matrix)
            centre = R.'*(c_G - tT); % distance of the end-effector position and centre position rotated for end-effector frame
            output = centre(1:2);
        end
    end
end
%------------- END OF CODE --------------
end
