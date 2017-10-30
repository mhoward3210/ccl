% Generates states (robot joint positions) and actions (robot joint velocities)
% and time for a circular wiping motion as unconstrained policy and planar surface
% constraints.
%   Saves the states and actions in file.
%
% Other m-files required: 
%   def_phi_4_cwm_sim.m
%   def_phib_5_spm_sim.m
%   def_phia_4_spm.m
%   def_constrained_policy.m

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 18-Oct-2017

%% User Input
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
NDem = 12; % number of demonstrations
Dangle = 30; % variation of angle in roll and pitch
% timming:
tf = 30; % duration of each simulation is seconds
freq = 30; % number of samples per second
cutOffTime = 7; % because the initial state of the simulation is not on the
% constraint, the simulation takes some time until the proportional
% controller converges the state to the constraint. This initial
% convergence time is cut out of the training data
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
x = cell(1, NDem); % configurations
u = cell(1, NDem); % actions
t = cell(1, NDem); % configuration data
p = cell(1, NDem); % configurations
c = cell(1, NDem); % wiping circle centre
r = cell(1, NDem); % wiping circle radious
n = cell(1, NDem); % planar surface normal
Phi = cell(1,NDem); % policy regressors
roll = cell(1,NDem);
pitch = cell(1,NDem);
unconstrainedPolicy = cell(1,NDem); % unconstrainedPolicy
time = linspace(0,tf,tf*freq); % time vector
timecond = time>cutOffTime;
for idx=1:NDem
    % Random variables:
    c{idx} = [rand().*0.15 + 0.45; rand().*0.1-0.05; rand().*0.15+0.35]; % generate random circle centre
    r{idx} = rand()*0.02+0.03; % generate random circle radious
    roll{idx} = rand()*(2*Dangle) - Dangle; 
    pitch{idx} = rand()*(2*Dangle) - Dangle;
    T = rpy2tr(roll{idx}, pitch{idx}, 0); % homogeneous transformation for the end-effector
    n{idx} = T(1:3,3);
    % Constant matrices:
    W_A = blkdiag(n{idx}.', n{idx}.', n{idx}.'); % constant gain matrix for the Constraint matrix
    W_b = -Kp*[W_A [-n{idx}.'*c{idx}; 0; 0]];
    % Definition of Constraint matrix and main task
    A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
    b = @(x) W_b*feval(Phi_b,x); % main task as a function of the configuration
    % Constrained Policie
    Phi{idx} = def_phi_4_cwm_sim(robot, c{idx}, r{idx}); % Get regressors for the unconstrained policy
    unconstrainedPolicy{idx} = @(x) Phi{idx}(x)*[1; 10];
    x_dot = def_constrained_policy(A, b, unconstrainedPolicy{idx});
    % solving motion
    sol = ode113(@(t,x) x_dot(x),[0 tf], x0);
    [traj, dtraj] = deval(sol,time); % evaluation of solution
    % store data
    x{idx} = num2cell(traj(:,timecond),1);
    u{idx} = num2cell(dtraj(:,timecond),1);
    timeprov = time(timecond); timeprov = timeprov - timeprov(1);
    t{idx} = num2cell(timeprov,1);
    % computation for plot purposes
    p{idx}=transl(robot.fkine(traj(:,timecond).'));
    disp(idx);
end
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
for idx=1:NDem
    % plot
    subplot(3,4,idx);
    plot3(c{idx}(1),c{idx}(2),c{idx}(3),'*g'); hold on;
    plot3(p{idx}(:,1),p{idx}(:,2),p{idx}(:,3),'*r');
    plotCircle3D(c{idx},r{idx},n{idx});
    xlabel('x'); ylabel('y'); zlabel('z');
    legend('centre','data','circle');
    axis equal;
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
