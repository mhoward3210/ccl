function functionHandle = getClosedFormNullSpaceProjectionMatrixEstimatior(Phi_A, Phi_b, ConstraintDim)
    functionHandle = @ClosedFormNullSpaceProjectionMatrixEstimatior;

    function [nullSpaceProjectionHat, WA_hat, Wb_hat] = ClosedFormNullSpaceProjectionMatrixEstimatior(q, u)
        % number of the regressors of the constraint function
        Ndof = length(q{1});
        NA = size(Phi_A(zeros(Ndof,1)),1);
        % Evaluate constraint and task regressors for all the data
        H = @(q,u) [Phi_A(q)*u; -Phi_b(q)];
        H_cell = cellfun(H, q, u, 'UniformOutput', false);
        H_matrix = cell2mat(H_cell);
        % Singular value decomposition to estimate the gains
        [U,~,~]=svd(H_matrix);
        W_hat = U(:,(end-(ConstraintDim-1)):end).'; % select the last ConstraintDim columns  
        WA_hat = W_hat(:,1:NA);
        Wb_hat = W_hat(:,(NA+1):end);
        % Definition of Constraint matrix and main task
        A_hat = @(q) WA_hat*Phi_A(q); % Constraint matrix as a function of configuration
        nullSpaceProjectionHat = def_null_space_proj(A_hat);
    end
end
