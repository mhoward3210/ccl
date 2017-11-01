% Given the data composed of states - joint positions - and actions - joint
% velocities, we estimate the null space projection matrix for each data 
% set/demonstration and use that result to compute the unconstrained policy.
% We then plot the result of the policy and estimated projection matrix with 
% the input data for the kuka end-effector cartesian positions.
%
% Other m-files required: 
%   def_phi_4_cwm.m
%   def_phib_4_spm_sim.m
%   def_phib_4_spm_exp.m
%   def_phia_4_spm.m
%   def_constrained_policy.m

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 23-Oct-2017

%% User Input
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
DH = [0.0, 0.31, 0.0, pi/2; % Robot Kinematic model specified by the Denavit-Hartnbergh
      0.0, 0.0, 0.0, -pi/2;
      0.0, 0.4, 0.0, -pi/2;
      0.0, 0.0, 0.0, pi/2;
      0.0, 0.39, 0.0, pi/2;
      0.0, 0.0, 0.0, -pi/2;
      0.0, 0.21-0.132, 0.0, 0.0];
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Add path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
addpath(genpath('../')); % add the library and it's subfolders to the path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Get data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Getting data ...\n');
%load('../data_generation/data_simulated.mat');
load('data.mat');
%load('demonstrations_mat/data.mat');
NDem = length(x); % number of demonstrations
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Initialize roobot model and the Regressors for the constraint and main task
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining robot model ...\n');
robot = SerialLink(DH); % Peters Cork robotics library has to be installed
Phi_A = def_phia_4_spm(robot); % Phi_A(x): vector of regressors for the Constraint matrix as a function of the configuration
Phi_b = def_phib_4_spm_exp(robot); % Phi_b(x): vector of regressors for the main task as a function of the configuration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Parallel computig settingsfeval(
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Initializing parallel pool ...\n');
gcp(); % Get the current parallel pool
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Define Policy Regressors for each demonstration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Defining Unconstrained Policy Regressors ...\n');
c = cell(1, NDem); % wiping circle centre
r = cell(1, NDem); % wiping circle radious
n = cell(1, NDem); % planar surface normal
p = cell(1, NDem); % end-effector cartesian position in global frame
Phi = cell(1,NDem);
getPos = @(q) transl(robot.fkine(q)); % compute end-effector postion
parfor idx=1:NDem
    %p{idx} = transl(robot.fkine(cell2mat(q{idx}).')); % compute end-effector postion
    p{idx} = getPos(cell2mat(x{idx}).'); % compute end-effector postion
    [c{idx}, r{idx}, n{idx}] = fit_3d_circle(p{idx}(:,1),p{idx}(:,2),p{idx}(:,3));
    Phi{idx} = def_phi_4_cwm(robot, c{idx}, r{idx}); % Get regressors for the unconstrained policy
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Estimate the null space projection matrix for each demonstration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Estimating constraints ...\n');
N_Estimator = getClosedFormNullSpaceProjectionMatrixEstimatior(Phi_A, Phi_b, 3);
N_hat = cell(1,NDem);
WA_hat = cell(1,NDem);
Wb_hat = cell(1,NDem);
parfor idx=1:NDem
    [N_hat{idx}, WA_hat{idx}, Wb_hat{idx}] = feval(N_Estimator, x{idx}, u{idx});
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model variance
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing data variance...\n');
xall = cell2mat([x{:}]).';
scale = 2;
model.var = scale.*std(xall,1,1).';
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model local Gaussian receptive fields centres'
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Receptive Fields Centres ...\n');
stream = RandStream('mlfg6331_64');  % Random number stream for parallel computation
options = statset('Display','off','MaxIter',200,'UseParallel',1,'UseSubstreams',1,'Streams',stream);
Nmodels = 25;
[~,C] = kmeans(xall,Nmodels,'Distance','cityblock','EmptyAction','singleton','Start','uniform',...
    'Replicates',10,'OnlinePhase','off','Options', options);
model.c = C.';
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Compute model parameters
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Learning Model Parameters...\n');
R_cell = cell(1,NDem);
parfor idx=1:NDem
    R_cell{idx} = compute_matrix_r(N_hat{idx}, Phi{idx}, x{idx});
end
R = cell2mat([R_cell{:}].');
Y = cell2mat([u{:}].');
B = zeros(size(Phi{1}(x{1}{1}),2),size(model.c,2));
w = @(m) @(x) exp(-0.5.*sum(bsxfun(@rdivide, bsxfun(@minus,x,model.c(:,m)).^2, model.var))).'; % importance weights W = [w1 w2 ... w_m ... w_M]
[nRrow,nRcol] = size(R_cell{1}{1});
parfor m=1:size(model.c,2)
    wm = feval(w, m);
    Wm = repelem(wm(xall.'),nRrow,nRcol);
    RWm = R.*Wm;
    B(:,m) = pinv(RWm.'*R)*RWm.'*Y;
end
model.b = B;
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Defining Unconstrained Policies
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Computing Unconstrained Policy...\n');
policy = cell(1,NDem);
parfor idx=1:NDem
    policy{idx} = def_weighted_linear_model(model, Phi{idx});
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%% Computing end-effector positions based on learned policies
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Compute End-Effector positions...\n');
Phi_b = def_phib_4_spm_sim(robot); % vector of regressors as a function of the configuration for the main task
pos = cell(1, NDem); % wiping circle centre
parfor idx=1:NDem
    % Problem specific constants taken from data:
    x0 = x{idx}{1}; % initial configuration
    Kp = 5; % proportional gain
    % Constant matrices:
    W_A = blkdiag(n{idx}.', n{idx}.', n{idx}.'); % constant gain matrix for the Constraint matrix
    W_b = -Kp*[W_A [-n{idx}.'*c{idx}; 0; 0]];
    % Definition of Constraint matrix and main task
    A = @(x) W_A*feval(Phi_A,x); % Constraint matrix as a function of configuration
    b = @(x) W_b*feval(Phi_b,x); % main task as a function of the configuration
    % Constrained Policie
    dx = def_constrained_policy(A, b, policy{idx});
    % solving motion
    [~,traj] = ode113(@(t,x) dx(x),[0 t{idx}{end}], x0);
    %pos=transl(robot.fkine(traj));
    pos{idx}=getPos(traj);
end
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
    plot3(c{idx}(1),c{idx}(2),c{idx}(3),'*r'); hold on;
    plot3(p{idx}(:,1),p{idx}(:,2),p{idx}(:,3),'g');
    plot3(pos{idx}(:,1),pos{idx}(:,2),pos{idx}(:,3));
    plotCircle3D(c{idx},r{idx},n{idx});
    xlabel('x'); ylabel('y'); zlabel('z');
    legend('centre','data','policy','circle');
    axis equal;
end
error('stop here');
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Auxiliar functions
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Deleting parallel pool...\n');
delete(gcp('nocreate'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
