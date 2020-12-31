% Given the data composed of states and actions, we estimate 
% the null space projection matrix for each data set/demonstration.
%
% Other m-files required: 
%   def_phia_4_spm.m
%   def_constraint_estimator.m

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% November 2017; Last revision: 03-Mar-2017

%% Add path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
addpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Get data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Getting data ...\n');
%load('../data_generation/data_simulated.mat');
load('data_2Dtoyexample.mat');
%load('demonstrations_mat/data.mat');
NDem = length(x); % number of demonstrations
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Estimate the null space projection matrix for each demonstration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
Phi_A = @(q) eye(2);
fprintf(1,'Estimating constraints ...\n');
N_Estimator = def_constraint_estimator(Phi_A,...
                         'system_type','stationary',...
                         'constraint_dim',1);
%--------------------------------------------------------------------------
% Set KCL settings for learning null space projection matrix:
%--------------------------------------------------------------------------
settings.dim_b          = 10 ;                          % dimensionality of the kernel function. 16 is normally enough for a 2D problem
settings.dim_n          = 20 ;                          % number of steps in each trajactory
settings.nTraj          = 50 ;                          % number of trajectories
settings.dim_exp        = 1 ;                           % number of experiment to repeat
settings.learn_alpha    = 1;
settings.dim_x          = 2 ;                                   % dimensionality of the state space
settings.dim_u          = 2 ;                                   % dimensionality of the action space
settings.dim_r          = 2 ;                                   % dimensionality of the task space
settings.dim_k          = 1 ;                                   % dimensionality of the constraint
settings.dt             = 0.1;                                  % time step
settings.null.alpha     = 0.5 ;                                   % null space policy scaling
%--------------------------------------------------------------------------
[N_hat, H_cell, W_hat] = N_Estimator(x, u);
NStr = cell2mat(u); % actions as matrix 7xN
Xtr = cell2mat(x); % state as matrix 7xN
model_alpha_ccl  = learn_alpha_ccl (NStr, Xtr, settings);
figure(1); quiver(Xtr(1,:),Xtr(2,:),NStr(1,:),NStr(2,:));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Plot null space projection matrix:
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf('True reference:\n');
disp(eye(2) - pinv(Aref)*Aref)
fprintf('KCL estimation:\n');
disp(model_alpha_ccl.f_proj(x{1}))
fprintf('UoE estimation:\n');
disp(N_hat(x{1}));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------