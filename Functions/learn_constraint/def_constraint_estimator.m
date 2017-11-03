function functionHandle = def_constraint_estimator(Phi_A, Phi_b, varargin)
    functionHandle = @ClosedFormNullSpaceProjectionMatrixEstimatior;
    
    % Default parameters:
    defaultConstraintDim = 1;
    defaultSystemType = 'forced_action';
    % List of expected types of constrained systems and respective vector H:
    expectedSystemTypes = {'forced_action','stationary'};
    expectedH = {@(q,u) [Phi_A(q)*u; -Phi_b(q)], @(q,u) Phi_A(q)*u};
    % Auxiliar functions:
    % Test if argument is a numeric, scaler, positive, and integer:
    validIntNum = @(x) isnumeric(x) && isscalar(x) && (x > 0) && rem(x,1)==0;
    % Define an input arguments parser:
    p = inputParser;
    p.CaseSensitive = true;
    p.FunctionName = 'def_constraint_estimator';
    % Add parameters to parser:
    addParameter(p, 'system_type', defaultSystemType,...
        @(x) any(validatestring(x,expectedSystemTypes)));
    addParameter(p, 'constraint_dim', defaultConstraintDim, validIntNum);
    % Parse function inputs:
    parse(p,varargin{:});
    % Get the constraint dimension and type of constrained system from
    % parser:
    ConstraintDim = p.Results.constraint_dim;
    systemType = p.Results.system_type;
    H = expectedH{strcmp(systemType,expectedSystemTypes)};
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
end
