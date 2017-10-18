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
Phi_A = getConstraintMatrixRegressor4SurfacePerpendicularMotion(robot); % Phi_A(q): vector of regressors for the Constraint matrix as a function of the configuration
Phi_b = getTaskRegressors4SurfacePerpendicultaMotionSimulated(robot); % Phi_b(q): vector of regressors for the main task as a function of the configuration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Generate data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Generating data ...\n');
NDem = 12; % number of demonstrations
q = cell(1, NDem); % configurations
u = cell(1, NDem); % actions
t = cell(1, NDem); % configuration data
p = cell(1, NDem); % configurations
c = cell(1, NDem); % wiping circle centre
r = cell(1, NDem); % wiping circle radious
n = cell(1, NDem); % planar surface normal
Phi = cell(1,NDem);
policy = cell(1,NDem);
Dangle = 20; % variation of angle in roll and pitch
getPos = @(q) transl(robot.fkine(q)); % compute end-effector postion
% timming:
tf = 30;
freq = 30;
time = linspace(0,tf,tf*freq); % time vector
timecond = time>7;
for idx=1:NDem
    % Random variables:
    c{idx} = [rand().*0.15 + 0.45; rand().*0.1-0.05; rand().*0.15+0.35]; % generate random circle centre
    r{idx} = rand()*0.02+0.03; % generate random circle radious
    roll = rand()*(2*Dangle) - Dangle; 
    pitch = rand()*(2*Dangle) - Dangle;
    T = rpy2tr(roll, pitch, 0); % homogeneous transformation for the end-effector
    n{idx} = T(1:3,3);
    % initial condition
    q0 = [0;0;0;pi/2;0;-pi/2;0];
    % proportional gain
    Kp = 5;
    % Constant matrices:
    W_A = blkdiag(n{idx}.', n{idx}.', n{idx}.'); % constant gain matrix for the Constraint matrix
    W_b = -Kp*[W_A [-n{idx}.'*c{idx}; 0; 0]];
    % Definition of Constraint matrix and main task
    A = @(q) W_A*feval(Phi_A,q); % Constraint matrix as a function of configuration
    b = @(q) W_b*feval(Phi_b,q); % main task as a function of the configuration
    % Constrained Policie
    Phi{idx} = getUnconstrainedPolicyRegressors4CircularWipingMotion(robot, c{idx}, r{idx}); % Get regressors for the unconstrained policy
    policy{idx} = @(q) Phi{idx}(q)*[1; 1];
    q_dot = getConstrainedPolicy(A, b, policy{idx});
    % solving motion
    sol = ode113(@(t,q) q_dot(q),[0 tf], q0);
    [traj, dtraj] = deval(sol,time); % evaluation of solution
    % store data
    q{idx} = num2cell(traj(:,timecond),1);
    u{idx} = num2cell(dtraj(:,timecond),1);
    timeprov = time(timecond); timeprov = timeprov - timeprov(1);
    t{idx} = num2cell(timeprov,1);
    % computation for plot purposes
    p{idx}=getPos(traj(:,timecond).');
    disp(idx);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Save data to file
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
save('data_simulated.mat','q','u','t');
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
    legend('centre','data','policy','circle');
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