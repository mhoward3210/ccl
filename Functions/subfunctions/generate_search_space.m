function search = generate_search_space (search)
% search = generate_search_space (search)
%
% Generate search space for theta and alpha
%
% Input:
%   search                                 Searching parameters
%
% Output:
%   search                                 Returned generated searching space

search.min_theta    = zeros(1,search.dim_t) ;
search.max_theta    = ( pi-(pi/search.num_theta) )*ones(1,search.dim_t) ;

num_theta  = search.num_theta ;
dim_u      = search.dim_u ;
dim_t      = search.dim_t ;
dim_s      = num_theta ^ dim_t ;
list       = zeros(dim_s, dim_t) ;
theta      = cell(1,dim_s) ;
alpha      = cell(1,dim_s) ;

for t = 1:dim_t
    list_theta  = linspace(search.min_theta(t), search.max_theta(t), num_theta) ;
    list_theta  = repmat(list_theta,    num_theta^(dim_t-t), 1) ;
    list(:,t)   = repmat(list_theta(:), num_theta^(t-1), 1) ;
end

% make list of alpha
for s = 1: dim_s
    theta{s} = list(s,:) ;
    alpha{s} = get_unit_vector_from_matrix ( list(s,:) ) ;
end
search.list = list ;
search.I_u      = eye(dim_u) ;
search.dim_s    = dim_s ;
search.theta    = theta ;
search.alpha    = alpha ;
search.interval = list(2,dim_t)-list(1,dim_t) ;
end
