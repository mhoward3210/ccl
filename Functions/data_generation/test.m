% GENERATE_DATA_KUKA_WIPING_MOTION.m - Generates states (robot
% joint positions) and actions (robot joint velocities) and time for a
% circular wiping motion as unconstrained policy and planar surface
% constraints.
%   Saves the states and actions in file.
%
% Other m-files required: 
%   getUnconstrainedPolicyRegressors4CircularWipingMotion.m
%   getTaskRegressors4SurfacePerpendicultaMotionSimulated.m
%   def_phia_4_spm.m
%   getConstrainedPolicy.m

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 18-Oct-2017

%% User Input
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% timming:
tf = 30; % duration of each simulation is seconds
cutOffTime = 0; % because the initial state of the simulation is not on the
% constraint, the simulation takes some time until the proportional
% controller converges the state to the constraint. This initial
% convergence time is cut out of the training data
freq = 30; % number of samples per second
x0 = [0;0;0;pi/2;0;-pi/2;0]; % initial condition
Kp = 5; % proportional gain
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Add path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
addpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Initialize roobot model and the Regressors for the constraint and main task
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining robot model ...\n');
DH = [0.0, 0.31, 0.0, pi/2; % Robot Kinematic model specified by the Denavit-Hartenberg
      0.0, 0.0, 0.0, -pi/2;
      0.0, 0.4, 0.0, -pi/2;
      0.0, 0.0, 0.0, pi/2;
      0.0, 0.39, 0.0, pi/2;
      0.0, 0.0, 0.0, -pi/2;
      0.0, 0.21-0.132, 0.0, 0.0];
robot = SerialLink(DH); % Peters Cork robotics library has to be installed
 % Phi_A(x): vector of regressors for the Constraint matrix as a function
 % of the state
Phi_A = def_phia_4_spm(robot);
% Phi_b(x): vector of regressors for the main task as a function of the
% state
Phi_b = def_phib_4_spm_sim(robot);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Generate data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Generating data ...\n');
time = linspace(0,tf,tf*freq); % time vector
timecond = time>cutOffTime;
% Random variables:
c = [0.55; 0.0; 0.6]; % generate random circle centre
r = 0.05; % generate random circle radious
roll = 30; 
pitch = -30;
%c = [0.59; 0.045; 0.4]; % generate random circle centre
% r = 0.047; % generate random circle radious
% roll = 14.443; 
% pitch = 24.83;
T = rpy2tr(roll, pitch, 0); % homogeneous transformation for the end-effector
n = T(1:3,3);
% Constant matrices:
W_A = blkdiag(n.', n.', n.'); % constant gain matrix for the Constraint matrix
W_b = -Kp*[W_A [-n.'*c; 0; 0]];
% Definition of Constraint matrix and main task
A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
b = @(x) W_b*feval(Phi_b,x); % main task as a function of the configuration
% Constrained Policie
Phi = getUnconstrainedPolicyRegressors4CircularWipingMotion(robot, c, r); % Get regressors for the unconstrained policy
unconstrainedPolicy = @(x) Phi(x)*[1; 10];
x_dot = getConstrainedPolicy(A, b, unconstrainedPolicy);
% solving motion
sol = ode113(@(t,x) x_dot(x),[0 tf], x0);
[traj, dtraj] = deval(sol,time); % evaluation of solution
% store data
x = num2cell(traj(:,timecond),1);
u = num2cell(dtraj(:,timecond),1);
timeprov = time(timecond); timeprov = timeprov - timeprov(1);
t = num2cell(timeprov,1);
% computation for plot purposes
p=transl(robot.fkine(traj(:,timecond).'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Save data to file
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
save('data_simulated.mat','x','u','t');
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot end-effector positions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Plotting Results...\n');
figure();
plot3(c(1),c(2),c(3),'*g'); hold on;
plot3(p(:,1),p(:,2),p(:,3),'*r');
plotCircle3D(c,r,n);
xlabel('x'); ylabel('y'); zlabel('z');
grid on;
legend('centre','data','circle');
axis square;
axis equal;
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
