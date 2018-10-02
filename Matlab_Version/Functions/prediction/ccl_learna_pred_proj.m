function Yp = ccl_learna_pred_proj (F,model)
% Yp = ccl_learna_pred_proj (F,model)
%
% Project a vector onto the image space of a learnt projection
%
% Input:
%
%    F                             Input vector before projection
%    model                         Learnt model for nullspace projection, containing fields
%         .P:                      Learnt projection matrix
%
% Output:
%
%    Yp                            The resulting vector after projection




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

Yp = model.P*F;
