function Unp = ccl_learnv_pred_ncl ( model, X )
% Unp = ccl_learnv_pred_ncl ( model, X )
%
% Predict null space components using learnt model
%
% Input:
%
%   model                       Learnt model parameters
%   X                           Input state variable
%
% Output:
%
%   Unp                          Null space component prediction




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

dim_N = size( X, 2 );
dim_U = size( model.w, 1 );
Unp   = zeros(dim_U, dim_N );
for i = 1 : dim_N
    Unp(:,i) = model.w * model.phi( X(:,i) );
end
end