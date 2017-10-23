function functionHandle = getNullSpaceProjection(constraintMatrix)
    functionHandle = @nullSpaceProjection;
    function output = nullSpaceProjection(q)
        A = constraintMatrix(q); % Compute constraint matrix for given configuration
        Ainv = pinv(A); % pseudo inverse of constraint matrix
        output = eye(length(q)) - Ainv*A; % Compute null-space projection matrix for given configuration
    end
end