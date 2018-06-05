function [nupe, v, mse] = ccl_error_nupe(F, Fp)
% [nupe, v, mse] = ccl_error_nupe(F, Fp)
%
% Calculate normalised unconstrained policy error (nUPE).  This quantity is
% used to evaluate the unconstrained learnt null space policy
%
% Input:                        
%
%   F                           True null space control commands
%   Fp                          Learnt null space control commands
%
% Output:
%   nupe                        Normalised unconstrained policy error (nUPE)
%   v                           Variance
%   mse                         Mean square error

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

[d v mse] = ccl_error_nmse(F,Fp);
nupe = sum(mse)/sum(v);

