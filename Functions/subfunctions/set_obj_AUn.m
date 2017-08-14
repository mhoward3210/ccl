function [fun] = set_obj_AUn (model, W, BX, RnUn)
% [fun] = obj_AUn (model, W, BX, RnUn)
% Objective funtion: minimise (A * Un)^2
%
% Input:
%   model                                Model related parameteres
%   W                                    Weights of parametric model
%   BX                                   Higher dimensional representation of X using gaussian kernel
%   RnUn                                 RnUn=Rn*Un
% Output:
%   fun                                  Returned objective function handle

dim_n   = size(BX,2) ;
W       = reshape(W, model.dim_u-model.dim_k, model.dim_b );
fun     = zeros(dim_n, 1) ;
theta   = [pi/2*ones(dim_n, (model.dim_k-1)), (W * BX)' ] ;
alpha   = get_unit_vector_from_matrix(theta) ;
for n   = 1 : dim_n
    fun(n)  = alpha(n,:) * RnUn(:,n) ;
end
end
