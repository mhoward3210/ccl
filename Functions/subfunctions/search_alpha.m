function model = search_alpha (BX, RnUn, model)
% model = search_alpha (BX, RnUn, model)
%
% Learning the constraint vector
%   Input:
%       BX                          Higher dimensional representation of X using gaussian kernel
%       RnUn                        RnUn=Rn*Un
%       model                       Model learnt from the last iteration
%   Output
%       model                       Updated model with the k^th constraint

%options.MaxFunEvals = 1e6 ;
options.MaxIter     = 1000 ;
options.TolFun      = 1e-6 ;
options.TolX        = 1e-6 ;
options.Jacobian    = 0 ;

obj                 = @(W) set_obj_AUn (model, W, BX, RnUn) ;   % setup the learning objective function
model.nmse          = 10000000 ;
for i = 1:1 % normally, the 1 attempt is enough to find the solution. Repeat the process if the process tends to find local minimum (i.e., for i = 1:5)
    W    = rand(1, (model.dim_u-model.dim_k)* model.dim_b) ; % make a random guess for initial value
    W    = solve_lm (obj, W, options );                      % use a non-linear optimiser to solve obj
    nmse = mean(obj(W).^2) / model.var ;
    fprintf('\t K=%d, iteration=%d, residual error=%4.2e\n', model.dim_k, i, nmse) ;
    if model.nmse > nmse
        model.w{model.dim_k}= reshape(W, model.dim_u-model.dim_k, model.dim_b) ;
        model.nmse          = nmse ;
    end
    if model.nmse < 1e-5 % restart a random initial weight if residual error is higher than 10^-5
        break
    end
end
end
