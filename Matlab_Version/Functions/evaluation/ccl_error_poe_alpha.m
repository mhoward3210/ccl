function [nPOE, vPOE, uPOE] = ccl_error_poe_alpha (f_proj, X, NS_t)
% [nPOE, vPOE, uPOE] = ccl_error_poe_alpha (f_proj, X, NS_t)
% Compute the normalised projection observation error (nPOE) with state dependant projections.
% To evaluate the fit without the true nullspace policy, an alternative criterion must be used.
% This indicates the quality of the learnt projection matrix in capturing the image space of
% the observations. i.e., how well the learnt N satisfy Nu = u
%
% Input:
%
%   f_proj                                   State dependent prediction of the projection matrix
%   X                                        True state
%   NS_t                                     True observed null-space component
%
% Output:
%
%   nPOE                                     Normalised projected observation error
%   vPOE                                     Variance of the nullspace policy
%   uPOE                                     Projected observation error




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

dim_n    = size(NS_t,2) ;
NS_p     = zeros(size(NS_t)) ;
for n = 1:dim_n
    NS_p(:,n) = f_proj(X(:,n)) * NS_t(:,n) ;
end
uPOE = mean (sum((NS_t-NS_p).^2,1) ) ;
vPOE = var(NS_t, 0, 2);
nPOE = uPOE / sum(vPOE);
end
