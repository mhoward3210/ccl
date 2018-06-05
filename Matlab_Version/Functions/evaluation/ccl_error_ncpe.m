function [ncpe, v, mse] = ccl_error_ncpe(F, Fp, P)
% [ncpe, v, mse] = ccl_error_ncpe(F, Fp, P)
%
% Calculate the normalised constrained policy error (nCPE).  This quantity
% is used to evaluate the quality of the leart policy under the same
% constrains
%
% Input:
%
%   F                           True Null space policy control commands
%   Fp                          Learnt Null space policy control commands
%   P                           Null space projection
%
% Output:
%   ncpe                        Normalised constrained policy error
%   v                           Variance in the true policy commands
%   mse                         Mean square error of the learnt policy




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

for n=1:size(P,3)
    Y (:,n) = P(:,:,n)*F (:,n);
    Yp(:,n) = P(:,:,n)*Fp(:,n);
end
[d1 d2 mse] = ccl_error_nmse(Y,Yp);
v    = var(F,0,2);         % get variance
ncpe = sum(mse)/sum(v);

