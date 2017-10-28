function functionHandle = getClosedFormNullSpaceProjectionMatrixEstimatior(Phi_A, Phi_b, ConstraintDim)
    functionHandle = @ClosedFormNullSpaceProjectionMatrixEstimatior;

    function [nullSpaceProjectionHat, WA_hat, Wb_hat] = ClosedFormNullSpaceProjectionMatrixEstimatior(q, u)
        % number of the regressors of the constraint function
        Ndata = length(q);
        Ndof = length(q{1});
        NA = size(Phi_A(zeros(Ndof,1)),1);
        Nb = size(Phi_b(zeros(Ndof,1)),1); 
        % Evaluate constraint and task regressors for all the data
        HH = zeros(NA+Nb,Ndata);
        parfor idx=1:Ndata
            % H(q,u): Matrix of regressors as a function of the configuration and action compute number of regressors
            HH(:,idx) = [feval(Phi_A, q{idx})*u{idx}; -feval(Phi_b,q{idx})];
        end
        % Singular value decomposition to estimate the gains
        [U,~,~]=svd(HH);
        W_hat = U(:,(end-(ConstraintDim-1)):end).'; % select the last ConstraintDim columns  
        WA_hat = W_hat(:,1:NA);
        Wb_hat = W_hat(:,(NA+1):end);
        % Definition of Constraint matrix and main task
        A_hat = @(q) WA_hat*Phi_A(q); % Constraint matrix as a function of configuration
        nullSpaceProjectionHat = getNullSpaceProjection(A_hat);
    end
end