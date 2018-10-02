function [optimal, result] = ccl_learna_nhat (Un)
% [optimal, result] = ccl_learna_nhat (Un)
%
% Learn state independent null space projection P
%
% Input:
%
%   Un                                      Observations of null space component
%
% Output:
%
%   optimal                                 Returned optimal model
%   result                                  Returned results




% CCL: A MATLAB library for Constraint Consistent Learning
% Copyright (C) 2007  Matthew Howard
% Contact: matthew.j.howard@kcl.ac.uk
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Library General Public
% License along with this library; if not, write to the Free
% Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

% setup parameters for search
search.dim_u    = size(Un,1) ;          % dimensionality of the action space
search.dim_n    = size(Un,2) ;          % number of data points
search.num_theta= 20 ;                 % number of candidate constraints (from 0 to pi)
search.dim_t    = search.dim_u - 1 ;    % number of parameters needed to represent an unit vector
search.epsilon  = search.dim_u * 0.001 ;
search          = ccl_math_ss (search) ;

% vectorises the matrix
Vn = zeros(search.dim_n, search.dim_u^2) ;
for n = 1:search.dim_n
    UnUn    = Un(:,n)*Un(:,n)' ;
    Vn(n,:) = UnUn(:)' ;
end

% search the first constraint
result.model{1} = ccl_learna_sfa (Vn, Un, [], search) ;
optimal         = result.model{1} ;

% search the next constraint until the new constraint does not fit
for alpha_id = 2:search.dim_t
    result.model{alpha_id} = ccl_learna_sa_nhat (Vn, Un, result.model{alpha_id-1}, search) ;
    if result.model{alpha_id}.nmse_j < .1
        optimal = result.model{alpha_id} ;
    else
        return ;
    end
end
end