function functionHandle = getUnconstrainedPolicyRegressors4CircularWipingMotion(robotHandle, c_G, radius)
    functionHandle = @Phi;
    function output = Phi(q)
        J = robotHandle.jacobe(q); % Robot Jacobian in the end-effector frame
        Jtask = J(1:2,:); % Jacobian for the x and y coordinates - perpendicular plane to the end-effector
        Phi_kappa = getPhi_kappa(robotHandle, c_G, radius); % regressors for the secondary task
        output = pinv(Jtask)*Phi_kappa(q);
    end
    function functionHandle = getPhi_kappa(robotHandle, c_G, radius)
        functionHandle = @Phi_kappa;
        function output = Phi_kappa(q)
            c_ro = feval(getC_ro(robotHandle, c_G),q); % centre of the circular motion to the end-effector relative to the end-effector frame
            c_ro_per = [0 -1; 1 0]*c_ro; % perpendicular to c_ro
            nc_ro = norm(c_ro); % total distance to the centre
            output = [c_ro_per c_ro*(1-(radius/nc_ro))];
        end
        function functionHandle = getC_ro(robotHandle, c_G)
            functionHandle = @c_ro;
            function output = c_ro(q)
                T = robotHandle.fkine(q); % end-effector homogeneous transformation
                tT = transl(T).'; % end-effector position
                R = t2r(T); % end-effector orientation (rotation matrix)
                centre = R.'*(c_G - tT); % distance of the end-effector position and centre position rotated for end-effector frame
                output = centre(1:2);
            end
        end
    end
end
