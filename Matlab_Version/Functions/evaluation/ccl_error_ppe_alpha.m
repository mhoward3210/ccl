function [nPPE, vPPE, uPPE] = ccl_error_ppe_alpha (f_proj, X_t, Pi, NS_t)
% [nPPE, vPPE, uPPE] = ccl_error_ppe_alpha (N_p, X_t, Pi, NS_t)
%
% Compute the normalised projected policy error (nPPE) with state dependant projection.
% This error measures the difference between the policy subject to the true constraints,
% and that of the policy subject to the estimated constraints.
%
% Input:
%
%   f_proj                                   State dependent prediction of the projection matrix
%   X_t                                      True state
%   NS_t                                     True null space component
%   Pi                                       True null space policy




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

dim_n = size(NS_t,2) ;
NS_p  = zeros(size(NS_t)) ;
for n = 1:dim_n
    NS_p(:,n) = f_proj(X_t(:,n)) * Pi(:,n) ;
end
uPPE = mean( sum( (NS_t-NS_p).^2, 1) );
vPPE = var(NS_t,0,2);
nPPE = uPPE / sum(vPPE);
end
