function [model, stats] = ccl_learna_sa_nhat (V, Un, model, search)
% [model, stats] = ccl_learna_sa_nhat (V, Un, model, search)
%
% Search the s-th constrain which orthogal to the previous ones
%
% Input:
%   V                                   Un*Un
%   Un                                  Null space component observations
%   model                               Model parameters
%   search                              Searching related parameters
%
% Output:
%   model                               Returned learnt alpha and performance
%   stats                               Mean square error

% for alpha_id > 1, check if alpha is orthogonal to the existing ones
for i = 1:search.dim_s
    abs_dot_product = abs ( model.alpha * search.alpha{i}' )  ; % dot product between this alpha and the previous one's
    if sum(abs_dot_product > 0.001) > 0 % ignore alpha that is not orthogonal to any one of them
        stats.umse(i) = 1000000000 ;
    else
        alpha         = [model.alpha; search.alpha{i}] ;
        AA            = pinv(alpha)*alpha ;
        stats.umse(i) = sum ( V*AA(:) ) ;
    end
end
[min_err, min_ind]  = min(stats.umse) ;
model.theta         = [ model.theta ; search.theta{min_ind} ] ;
model.alpha         = [ model.alpha ; search.alpha{min_ind} ] ;
model.P             = search.I_u - pinv(model.alpha) * model.alpha ;
model.variance      = sum(var( model.P*Un, 0, 2)) ;
model.umse_j        = stats.umse(min_ind) / search.dim_n ;
model.nmse_j        = model.umse_j        / model.variance ;
end
