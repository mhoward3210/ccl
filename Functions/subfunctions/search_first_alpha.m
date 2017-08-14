function [model, stats] = search_first_alpha (V, Un, model, search)
% [model, stats] = search_first_alpha (V, Un, model, search)
%
% Search the first dimension of constraint matrix alpha
%
% Input:
%   V                                   UnUn(:)'
%   Un                                  Observations of null space component
%   model                               Model related parameters
%   search                              Searching configurations
%
% Output:
%   model                               Returned model
%   stats                               Mean

for i = 1:search.dim_s
    alpha         = search.alpha{i} ;
    AA            = pinv(alpha)*alpha ;
    stats.umse(i) = sum ( V*AA(:) ) ;
end
[min_err, min_ind]  = min(stats.umse) ;
model.theta         = search.theta{min_ind} ;
model.alpha         = search.alpha{min_ind} ;
model.P             = search.I_u - pinv(model.alpha) * model.alpha ;
model.variance      = sum(var( model.P*Un, 0, 2)) ;
model.umse_j        = stats.umse(min_ind) / search.dim_n ;
model.nmse_j        = model.umse_j        / model.variance ;
end
