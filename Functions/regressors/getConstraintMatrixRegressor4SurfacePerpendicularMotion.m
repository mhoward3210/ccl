% Get regressors for the constraint matrix when representing the constraint
% matrix as the product of a vector of regressors by a constant matrix
function functionHandle = getConstraintMatrixRegressor4SurfacePerpendicularMotion(robotHandle)
    functionHandle = @Phi_A;
    function output = Phi_A(q)
        J = robotHandle.jacob0(q); % Robot Jacobian in the global reference frame
        JtT = J(1:3,:); % Jacobian for the end-effector position
        Jrot = J(4:6,:); % rotation component of Jacobian
        rot = t2r(robotHandle.fkine(q)); % end-effector orientation (rotation matrix)
        xT = rot(:,1); yT = rot(:,2); % Unit vectors that define the plane perpendicular to end-effector
        JxT = -skew(xT)*Jrot; JyT = -skew(yT)*Jrot; % Jacobians for the end-effector frame unit vectors
        output = [JtT; JxT; JyT];
    end
end
