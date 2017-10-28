% Get constrained policy as function of configuration given the follow .  ing
% functions: constraintMatrix, task, unconstrained policy
function functionHandle = getConstrainedPolicy(constraintMatrix, task, unconstrainedPolicy)
    functionHandle = @constrainedPolicy; % return handle of constrained policy 
    function output = constrainedPolicy(q)
        A = constraintMatrix(q); % Compute constraint matrix for given configuration
        Ainv = pinv(A); % pseudo inverse of constraint matrix
        N = eye(length(q)) - Ainv*A; % Compute null-space projection matrix for given configuration
        b = task(q); % Compute task vector for given configuration
        pi = unconstrainedPolicy(q); % Compute unconstrained policy
        output = Ainv*b + N*pi; % output constrained policy
    end
end