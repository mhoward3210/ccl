function [fun J] = ccl_obj_ncl (model, W, BX, U)
% [fun J] = ccl_obj_ncl (model, W, BX, U)
%
% Learning objective function for learning nullspace component. The goal is
% to model the nullspace component Uns such that the difference between P*U
% and the learnt Uns is minimised. P is a projection matrix that projects U
% onto the image space of Uns.
%
% Input:
%
%    model                              Current model
%    W                                  Weights
%    BX                                 B(X)
%    U                                  Observed actions
%
% Output:
%
%    fun                                Error function of weight W
%    J                                  Jacobian




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

[dim_U, dim_N ] = size(U) ;
lambda  = 1e-8;
W       = reshape(W, dim_U, model.num_basis );
J       = zeros( dim_N, model.num_basis * dim_U);
fun     = zeros( dim_N, 1 );

for n = 1 : dim_N
    b_n     = BX(:,n);
    u_n     = U (:,n);
    Wb      = W    * b_n;
    c       = Wb'  * Wb;
    a       = u_n' * Wb;
    j_n     = ( u_n*b_n'*c - Wb*b_n'*(c+a - 2*lambda) ) / (sqrt(c)*c);
    J(n,:)  = j_n(:);
    fun(n)  = (a-c + lambda)/sqrt(c);
end
end
