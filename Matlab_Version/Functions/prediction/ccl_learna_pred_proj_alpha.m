function N = ccl_learna_pred_proj_alpha (model, q, Iu)
% N = ccl_learna_pred_proj_alpha (model, q, Iu)
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

Rn = Iu ;                               % initial rotation
A  = zeros(model.dim_k, model.dim_u) ;  % Initial constraint matrix
bx = model.phi(q) ;                     % gaussian kernel of q
for k = 1:model.dim_k
    theta   = [ pi/2 * ones(1,k-1), (model.w{k} * bx )' ] ; % the kth constraint parameter
    A(k,:)  = ccl_math_uvec(theta) *  Rn ;    % the kth constraint vector
    Rn      = ccl_math_rotmat (theta, Rn, model, k) ;   % update rotation matrix for (k+1)
end
N = Iu - A'*A ;
end
