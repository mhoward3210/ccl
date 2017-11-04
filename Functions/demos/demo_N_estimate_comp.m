% Given the data composed of states - joint
% positions - and actions - joint velocities, we estimate the null space
% projection matrix for each data set/demonstration.
%
% Other m-files required: 
%   def_phia_4_spm.m
%   def_constraint_estimator.m

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% November 2017; Last revision: 04-Nov-2017

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
addpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Get data
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Getting data ...\n');
%load('../data_generation/data_simulated.mat');
load('data_simulated.mat');
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
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Estimate the null space projection matrix for each demonstration
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
fprintf(1,'Estimating constraints ...\n');
N_Estimator = def_constraint_estimator(Phi_A,...
                         'system_type','stationary',...
                         'constraint_dim',3);
N_hat = cell(1, NDem);
H_cell = cell(1, NDem);
W_hat = cell(1, NDem);
%--------------------------------------------------------------------------
% Set KCL settings for learning null space projection matrix:
%--------------------------------------------------------------------------
model_lambda_ccl = cell(1, NDem);
settings.dim_x = 7; % dimensionality of the state space
settings.dim_u = 7;% dimensionality of the action space
settings.dim_r = 3; % dimensionality of the task space
settings.dim_b = 10; % dimensionality of the kernal
%--------------------------------------------------------------------------
for idx=1:1
    [N_hat{idx}, H_cell{idx}, W_hat{idx}] = N_Estimator(x{idx}, u{idx});
    NStr = cell2mat(u{idx}); % actions as matrix 7xN
    Xtr = cell2mat(x{idx}); % state as matrix 7xN
    J = Phi_A; % Regressors for constraint matrix: A = lambda * J;
    model_lambda_ccl{idx}  = learn_lambda_ccl (NStr, Xtr, J, settings);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Remove path
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
rmpath(genpath('../'));
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------