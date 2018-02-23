%% Add path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
addpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% demo_toy_example_2D
% This demo demonstrates a toy 2D problem for the usage of the CCL
% library.;                                  >\r');
%% GENERATIVE MODEL PARAMETERS
ctr = .9*ones(1,3); lwtr = 1;                                   % colour, linewidth for training data
cv  = [0  0  0];    lwv  = 1;                                   % colour, linewidth for visualisation data
cpv = [1 .5 .5];    lwpv = 2;                                   % colour, linewidth for visualisation data predictions
settings.dim_x          = 2 ;                                   % dimensionality of the state space
settings.dim_u          = 2 ;                                   % dimensionality of the action space
settings.dim_r          = 2 ;                                   % dimensionality of the task space
settings.dim_k          = 1 ;                                   % dimensionality of the constraint
settings.dt             = 0.1;                                  % time step
settings.null.alpha     = 0.5 ;                                   % null space policy scaling
settings.s2y  = .01;                                                     % noise in output
xmax = ones(settings.dim_x,1); xmin=-xmax;                                % range of data

settings.projection = 'state_independant';                        % {'state_independant' 'state_dependant'}
settings.task_policy_type = 'random';                           % {'random'}
settings.null_policy_type = 'limit_cycle';                 % {'limit_cycle' 'linear_attractor' 'linear'}
settings.control_space    = 'end_effector';                     % control space in end_effector

fprintf('< Dim_x             = %d                                                   >\r',settings.dim_x);
fprintf('< Dim_u             = %d                                                   >\r',settings.dim_u);
fprintf('< Dim_r             = %d                                                   >\r',settings.dim_r);
fprintf('< Dim_k             = %d                                                   >\r',settings.dim_k);
fprintf('< Constraint        = %s                                                   >\r',settings.projection);
fprintf('< Null_policy_type  = %s                                                   >\r',settings.null_policy_type);
fprintf('< Task_policy_type  = %s                                                   >\r',settings.task_policy_type);

%% NULL SPACE POLICY GENERATION
switch settings.null_policy_type
    case 'limit_cycle'
        radius = 0.75;                                          % radius of attractor
        qdot   = 1.0;                                           % angular velocity
        w      = 1.0;                                           % time scaling factor
        f_n = @(x)(-w*[(radius-x(1,:).^2-x(2,:).^2).*x(1,:) - x(2,:)*qdot;(radius-x(1,:).^2-x(2,:).^2).*x(2,:) + x(1,:)*qdot]);
    case 'linear_attractor'
        target = [0 0]';
        f_n    = @(x) settings.null.alpha .* (target - x);      % nullspace policy
    case 'linear'
        w    = [1 2;3 4;-1 0];
        f_n    = @(x)(([x;ones(1,size(x,2))]'*w)');
end
settings.f_n = f_n;

%% TASK SPACE POLICY GENERATION
f_b = @(N)(2*rand(1,1,N)-1);
% f_b    = @(x) settings.null.alpha .* (randi([-2,2]) - x);      % nullspace policy
settings.f_b = f_b;

%% Joao: Here is where you are REALLY generating the data:
model = [];
Ntr  = 500;
Nte   = 500;
A_ = orth( rand(2, 1) )';
f_A = @(q)(A_);                       % random constraint
model.num_basis = 16 ;                % define the number of radial basis functions
settings.f_A = f_A;
settings.grid_on = 1;
fprintf('\n< Generating training dataset for learning null space components  ...     >\r');
settings.N = Ntr;
Dtr = generate_data_ccl(settings);

%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------