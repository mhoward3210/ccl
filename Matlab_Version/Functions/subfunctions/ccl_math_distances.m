function D = ccl_math_distances (A, B)
% D = ccl_math_distances (A,B)
%
% calculate the euclidean distance between two datasets A and B
%
% input:
%
%    A: a dataset with N data points
%    B: a dataset with M data points
%
% output:
%
%    D: euclidean distance between A and B. D is a MxN matrix





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

    numA= size(A,2) ;
    numB= size(B,2) ;
    A2  = sum(A.^2) ;
    B2  = sum(B.^2) ;    
    D   = repmat( A2',1, numB ) + repmat( B2, numA, 1) - 2*A'*B ;   
end
