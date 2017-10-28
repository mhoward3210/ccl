% Get regressors for the main task when representing the task vector as a
% product of a vector of regressors by a constant matrix
function functionHandle = getTaskRegressors4SurfacePerpendicularMotionExp(robotHandle)
    functionHandle = @Phi_b;
    function output = Phi_b(q)
        T = robotHandle.fkine(q); % end-effector homogeneous transformation
        tT = reshape(transl(T),[],1); % end-effector position
        rot = t2r(T); % end-effector orientation (rotation matrix)
        xT = rot(:,1); yT = rot(:,2); % Unit vectors that define the plane perpendicular to end-effector
        q_M = q*q.'; % Matrix with the second order binomials of the configuration
        mask = triu(true(length(q))); % mask to choose halp
        q_2order = q_M(mask); % second order binomials of the configuration
        output = [q_2order; q; tT; xT; yT; 1];
    end
end