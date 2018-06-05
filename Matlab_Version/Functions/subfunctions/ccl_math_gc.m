function centres = ccl_math_gc(X, num_basis )
% centres = ccl_math_gc(X, num_basis )
%
% Generate the centres of the basis functions uniformly distributed over
% the range of the input data X
%
% Input:
%
%   X                       Input data
%   num_basis               Number of basis functions
%
% Output:
%
%   centres                 Centres of the basis functions




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

    if ~exist('num_basis')    
        n = sqrt(floor(sqrt(size(X,2)/6))^2) ;
    else
        n = floor(sqrt(num_basis));
    end           
    xmin = min(X,[],2);
    xmax = max(X,[],2);

    % allocate centres on a grid
    [xg,yg] = meshgrid( linspace(xmin(1)-0.1,xmax(1)+0.1,n),...
                        linspace(xmin(2)-0.1,xmax(2)+0.1,n));
    centres = [xg(:) yg(:)]';    
end

