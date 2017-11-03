function functionHandle = def_constraint_estimator(Phi_A, Phi_b, varargin)
    functionHandle = @ClosedFormNullSpaceProjectionMatrixEstimatior;
    
    validIntNum = @(x) isnumeric(x) && isscalar(x) && (x > 0) && rem(x,1)==0;
    
    defaultConstraintDim = 1;
    defaultSystemType = 'forced_action';
    expectedSystemTypes = {'forced_action','stationary'};
    expectedH = {@(q,u) [Phi_A(q)*u; -Phi_b(q)], @(q,u) Phi_A(q)*u};

    p = inputParser;
    p.CaseSensitive = true;
    p.FunctionName = 'def_constraint_estimator';
    addParameter(p, 'system_type', defaultSystemType,...
        @(x) any(validatestring(x,expectedSystemTypes)));
    addParameter(p, 'constraint_dim', defaultConstraintDim, validIntNum);
    parse(p,varargin{:});
    
    ConstraintDim = p.Results.constraint_dim;
    systemType = p.Results.system_type;
    H = expectedH{strcmp(systemType,expectedSystemTypes)};

    function [nullSpaceProjectionHat, H_cell, W_hat] = ClosedFormNullSpaceProjectionMatrixEstimatior(q, u)
        % Evaluate H regressors for given the data set
        H_cell = cellfun(H, q, u, 'UniformOutput', false);
        H_matrix = cell2mat(H_cell);
        % Singular value decomposition to estimate the gains
        [U,~,~]=svd(H_matrix);
        W_hat = U(:,(end-(ConstraintDim-1)):end).'; % select the last ConstraintDim columns 
        NA = size(Phi_A(q{1}),1); % number of the regressors for the constraint matrix
        WA_hat = W_hat(:,1:NA);
        % Definition of Constraint matrix
        A_hat = @(q) WA_hat*Phi_A(q); % Constraint matrix as a function of configuration
        nullSpaceProjectionHat = def_null_space_proj(A_hat);
    end
end
