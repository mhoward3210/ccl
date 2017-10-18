% Get regressors for the main task when representing the task vector as a
% product of a vector of regressors by a constant matrix
function functionHandle = getTaskRegressors4SurfacePerpendicultaMotionSimulated(robotHandle)
    functionHandle = @Phi_b;
    function output = Phi_b(q)
        T = robotHandle.fkine(q); % end-effector homogeneous transformation
        tT = transl(T).'; % end-effector position
        rot = t2r(T); % end-effector orientation (rotation matrix)
        xT = rot(:,1); yT = rot(:,2); % Unit vectors that define the plane perpendicular to end-effector
        output = [tT; xT; yT; 1];
    end
end
