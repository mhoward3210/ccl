function [umse, variance, nmse] = ccl_error_npe (U, Unp)
% [umse, variance, nmse] = ccl_error_npe (U, Unp)
%
% Calculate the nullspace projection error(NPE). This quantity is used to evaluate
% the quality of the learnt model in the absence of the true nullspace components
%
% Input:
%
%   U                           True observations
%   Unp                         Unp: learnt nullspace component
%
% Output:
%
%   umse                        Mean square error
%   variance                    variance in the observations
%   nmse                        Normalised mean square error




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

N   = size(U,2);
umse= 0;

for n=1:N
    up      = Unp(:,n);
    P       = up*up'/(up'*up);
    Pu      = P*U(:,n);
    umse    = umse + norm(Pu - up)^2;
end

umse    = umse / N ;
variance= sum(var(U,1,2)) ;
nmse    = umse / variance  ;
end
