function N = ccl_learna_pred_proj_lambda (q, model, J, Iu)
% N = ccl_learna_pred_proj_lambda (model, q, Iu)
% Prediction of the projection matrix
% Our model predicts the constraint parameters. this function is used to
% reconstuct the projection matrix from constraint paramters.
%
% Input:
%   model                                   Parametric model parameters
%   q                                       Joint state data
%   Iu                                      Identity matrix
%
% Output:
%   N                                       Null space projection

Rn      = Iu ;                                  % Initial rotation matrix
Lambda  = zeros(model.dim_k, model.dim_r) ;     % Initial selection matrix

for k = 1:model.dim_k
    theta       = [pi/2 * ones(1,k-1) ,  (model.w{k} * model.phi(q) )' ] ;
    alpha       = ccl_math_uvec(theta) ;                      % the kth alpha_0
    Lambda(k,:) = alpha * Rn ;                                  % rotate alpha_0 to get the kth constraint
    Rn          = ccl_math_rotmat (theta, Rn, model, k) ;   % update rotation matrix for (k+1)
end
A = Lambda * J(q) ;
N = eye(model.dim_u) - pinv(A)*A ;
end
