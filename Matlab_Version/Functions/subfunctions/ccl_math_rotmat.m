function R = ccl_math_rotmat(theta_k, current_Rn, search, alpha_id)
% R = ccl_math_rotmat(theta_k, current_Rn, search, alpha_id)
%
% Calculate rotation matrix after finding the k^th constraint vector.
% The result is rotation matrix that rotate vectors into a space orthogonal to all constraint vectors
%
% Input:
%   theta_k                               Theta from kth constrains
%   current_Rn                            Rotation matrix from the last iteration
%   search                                Dimensionality related searching parameters
%   alpha_id                              Current searching dimension of alpha
%
% Output:
%   R                                     Rotation matrix

R = eye(search.dim_u) ;
for d = alpha_id : search.dim_t
    R = R * ccl_math_mgmat(search.dim_u, d, d+1, theta_k(d) )' ;
end
R = R * current_Rn ;
end
