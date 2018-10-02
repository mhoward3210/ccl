function [nPOE, vPOE, uPOE] = ccl_error_poe(U_t, N_p, Pi)
% [nPOE, vPOE, uPOE] = ccl_error_poe(U_t, N_p, Pi)
%
% Compute the normalised projected observation error (nPOE). To evaluate the fit without the
% true nullspace policy, an alternative criterion must be used. This indicates
% the quality of the learnt projection matrix in capturing the image space of
% the observations. i.e., how well the learnt N satisfy Nu = u
%
% Input:
%
%    U_t                                    True observation
%    N_p                                    Learnt projection matrix
%    P_i                                    True nullspace policy
%
% Output:
%
%    nPPE                                   Normalised projected observation error
%    vPPE                                   Variance of the nullspace policy
%    uPPE                                   Projected observation error




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
    U_p(:,n) = N_p*U_t(:,n) ;
end
uPOE = sum((U_t-U_p).^2,2) / dim_n ;
vPOE = var(Pi, 0, 2);
nPOE = sum(uPOE) / sum(vPOE);
end
