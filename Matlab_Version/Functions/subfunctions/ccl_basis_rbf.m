function BX = ccl_basis_rbf ( X, C, s2 )
% BX = ccl_basis_rbf ( X, C, s2 )
%
% Calculate normalised Gaussian radial basis function of input X

% Input:
%
%   X                              Input data
%   C                              Centre
%   s2                             Variance
%
% Output:
%
%   BX                             Normalised gaussian radial basis function of X




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

D   = ccl_math_distances(C,X) ;                              % distance between C and X
BX  = exp(-0.5/s2*D) ;                              % radial basis function of X
BX  = BX.*repmat(sum(BX).^(-1),size(C,2),1);  % normalise
end
