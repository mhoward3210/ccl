function functionHandle = def_constraint_estimator(Phi_A, Phi_b, varargin)
    functionHandle = @ClosedFormNullSpaceProjectionMatrixEstimatior;
    
    validIntNum = @(x) isnumeric(x) && isscalar(x) && (x > 0) && rem(x,1)==0;
    
    defaultConstraintDim = 1;
    defaultSystemType = 'forced_action';
    expectedSystemTypes = {'forced_action','stationary'};

    p = inputParser;
    p.CaseSensitive = true;
    p.FunctionName = 'def_constraint_estimator';
    addParameter(p, 'system_type', defaultSystemType,...
        @(x) any(validatestring(x,expectedSystemTypes)));
    addParameter(p, 'constraint_dim', defaultConstraintDim, validIntNum);
    parse(p,varargin{:});
    
    ConstraintDim = p.Results.constraint_dim;
    %SystemType = p.Results.system_type;

    function [nullSpaceProjectionHat, H_cell, W_hat] = ClosedFormNullSpaceProjectionMatrixEstimatior(q, u)
        % number of the regressors for the constraint matrix
        NA = size(Phi_A(q{1}),1);
        % Evaluate constraint and task regressors for all the data
        H = @(q,u) [Phi_A(q)*u; -Phi_b(q)];
        H_cell = cellfun(H, q, u, 'UniformOutput', false);
        H_matrix = cell2mat(H_cell);
        % Singular value decomposition to estimate the gains
        [U,~,~]=svd(H_matrix);
        W_hat = U(:,(end-(ConstraintDim-1)):end).'; % select the last ConstraintDim columns  
        WA_hat = W_hat(:,1:NA);
        % Definition of Constraint matrix and main task
        A_hat = @(q) WA_hat*Phi_A(q); % Constraint matrix as a function of configuration
        nullSpaceProjectionHat = def_null_space_proj(A_hat);
    end
end
