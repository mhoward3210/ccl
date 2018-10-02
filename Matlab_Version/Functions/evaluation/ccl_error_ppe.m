function [nPPE, vPPE, uPPE] = ccl_error_ppe(U_t, N_p, Pi)
% [nPPE, vPPE, uPPE] = ccl_error_ppe(U_t, N_p, Pi)
%
% Compute the normalised projected policy error (nPPE). This error measures the
% difference between the policy subject to the true constraints, and that of
% the policy subject to the estimated constraints.
%
% Input:
%
%   U_t                              True observation
%   N_p                              Learnt projection matrix
%   P_i                              True nullspace policy
%
% Output:
%
%   nPPE                             Normalised projected policy error
%   vPPE                             Variance of the nullspace policy
%   uPPE                             Projected policy error




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

dim_n   = size(U_t,2) ;
U_p     = zeros(size(U_t)) ;
for n = 1:dim_n
    U_p(:,n) = N_p*Pi(:,n) ;
end
uPPE = sum((U_t-U_p).^2,2) / dim_n ;
vPPE = var(Pi,0,2);
nPPE = sum(uPPE) / sum(vPPE);
end
