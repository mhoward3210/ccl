function functionHandle = def_constraint_estimator(Phi_A, varargin)
% Defines a closed form null space constraint projection estimator.
%
% Consider a stationary constrained Pfaffian system modelled as:
%
%         A(x) * u(x) = 0,
%
% or a forced action constrained Pfaffian system modelled as:
%
%         A(x) * u(x) = b(x),
%
% where A(x) is the constraint matrix, x is the state, u(x) are the constrained actions, and
% b(x) is a task policy that ensures the system satisfies the constraint.
% Consider we model A(x) and b(x) as linear combinations of a set of regressors that depend on the state x:
%
%         A(x) = W_A * Phi_A(x),
%
%         b(x) = W_b * Phi_b(x),
%
% where W_b is a matrix of weights, and Phi_b(x) is a matrix of regressors, and likewise to W_A and Phi_A.
% This regressors are a function of the robot configuration - column vector.
% 
% def_constraint_estimator returns a MatLab function handle to an estimator function.
% This estimator gets as input a set of states (stored in a cell) and a set of actions (stored in a cell)
% and returns the null space projection matrix function for the data set constraint.
%
% This closed form null space projection estimator works by constructing the following system:
%
%         W * H = 0,
%
% and estimating W using svd decomposition of the matrix H.
% Each column of the matrix H is the evaluation of the regressors for each data point (state, action) as:
%
%         Hi(x,u) = Phi_A(x)*u, for stationary system, and
%
%         Hi(x,u) = [Phi_A(x)*u; -Phi_b(x)], for forced action systems.
%
% For details please refer to:
%   Leopoldo Armesto, João Moura, Vladimir Ivan, Antonio Salas, and Sethu Vijayakumar.
%   Learning Constrained Generaliz- able Policies by Demonstration.
%   In Proceedings of Robotics: Science and System XIII, 7 2017.
%
% Syntax:  
%     functionHandle = def_constraint_estimator(Phi_A)
%     functionHandle = def_constraint_estimator(__, name, value)
%
% Description:
%     functionHandle = def_constraint_estimator(Phi_A) returns the estimator for the default case of stationary system, 1 constraints.
%
%     functionHandle = def_constraint_estimator(__, name, value), specifies additional parameters for the estimator construction.
%
% Inputs:
%     Phi_A - MatLab function handle to constraint Matrix regressors;
%     'system_type', type - where type can be 'stationary' (default) or 'forced_action';
%     'task_regressors', phi_b - where phi_b is a MatLab function handle to the task function of the system state;
%     'constraint_dim', int_number - where int_number is an positive integer number indicating the number of constraints.
%
% Outputs:
%     functionHandle - MatLab function handle with cell data set of states and a cell data set of
%                      actions as inputs. This function handle is a null space projection matrix
%                      estimator, whose output is a MatLab function handle to a null space 
%                      projection matrix estimate function, the matrix H used to do the 
%                      estimate, and the estimated weights W.
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
%     % Phi_b(x): vector of regressors for the main task as a function of the
%     % state:
%     Phi_b = def_phib_4_spm_sim(robot);
%     % Define constraint estimator:
%     N_Estimator = def_constraint_estimator(Phi_A,...
%                        'system_type','forced_action',...
%                        'task_regressors',Phi_b,...
%                        'constraint_dim',3);
%     % Estimate null space projection matrix of constraint given dataset (x,u):
%     [N_hat, H_cell, W_hat] = N_Estimator(x, u); % Where x and u are cells containing column vectors.
%
%------------- BEGIN CODE --------------
functionHandle = @ClosedFormNullSpaceProjectionMatrixEstimatior;
% Default parameters:
defaultConstraintDim = 1;
defaultSystemType = 'stationary';
% List of expected types of constrained systems and respective vector H:
expectedSystemTypes = {'forced_action','stationary'};

% Auxiliar functions:
% Test if argument is a numeric, scaler, positive, and integer:
validIntNum = @(x) isnumeric(x) && isscalar(x) && (x > 0) && rem(x,1)==0;
validFunc = @(f) isa(f, 'function_handle');
% Define an input arguments parser:
p = inputParser;
p.CaseSensitive = true;
p.FunctionName = 'def_constraint_estimator';
% Add required variables:
addRequired(p,'Phi_A',validFunc);
% Add parameters to parser:
addParameter(p, 'system_type', defaultSystemType,...
    @(x) any(validatestring(x,expectedSystemTypes)));
addParameter(p, 'constraint_dim', defaultConstraintDim, validIntNum);
addParameter(p, 'task_regressors', [], validFunc);
% Parse function inputs:
parse(p,Phi_A,varargin{:});
% Get the constraint dimension, type of constrained system, and first
% task regressors from parser:
ConstraintDim = p.Results.constraint_dim;
systemType = p.Results.system_type;
if strcmp(systemType, 'stationary')
    H = @(q,u) Phi_A(q)*u;
elseif strcmp(systemType, 'forced_action')
    if validFunc(p.Results.task_regressors)
        Phi_b = p.Results.task_regressors;
        H = @(q,u) [Phi_A(q)*u; -Phi_b(q)];
    else
        error('def_constraint_estimator: ''task_regressors'' parameter not defined');
    end
end
% Returned function:
function [nullSpaceProjectionHat, H_cell, W_hat] = ClosedFormNullSpaceProjectionMatrixEstimatior(q, u)
    % Evaluate H regressors for the given data set
    H_cell = cellfun(H, q, u, 'UniformOutput', false);
    H_matrix = cell2mat(H_cell);
    % Singular value decomposition to estimate the gains
    [U,~,~]=svd(H_matrix);
    W_hat = U(:,(end-(ConstraintDim-1)):end).'; % Select the last ConstraintDim columns.
    % Select first NA columns in case of using some constrained action:
    NA = size(Phi_A(q{1}),1); % Number of the regressors for the constraint matrix.
    WA_hat = W_hat(:,1:NA);
    % Definition of Constraint matrix and null space projection:
    A_hat = @(q) WA_hat*Phi_A(q); % Constraint matrix as a function of configuration.
    nullSpaceProjectionHat = def_null_space_proj(A_hat);
end
%------------- END OF CODE --------------
end
