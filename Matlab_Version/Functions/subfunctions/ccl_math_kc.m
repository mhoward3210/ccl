function [M,dist_new] = ccl_math_kc (X, K)
% [M,dist_new] = ccl_math_kc (X, K)
%
% Initialisation: randomly select K points as the centre
%
% Input:
%
%   X                   Input signals
%   K                   Number of Gaussian distributions
%
% Output:
%
%   M                   Mean value of the K Gaussians
%   dist_new            sum of overall distance in between K clusters and
%                       the input data set




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

N        = size(X,2) ;
ind      = randperm(N) ;
ind      = ind(1:K) ;
M        = X(:,ind) ;
dist_old = realmax ;

for iter= 0:1000
    D             = ccl_math_distances(M,X);
    [mD,ind]      = min(D);
    emptyClusters = [];
    
    for k = 1 : K
        ix = find(ind == k);
        if ~isempty(ix)
            M(:,k) = mean(X(:,ix),2);
        else
            emptyClusters = [emptyClusters k];
        end
    end
    
    dist_new = sum(mD);
    
    if isempty (emptyClusters)
        if abs (dist_old-dist_new) < 1e-10;
            return;
        end
    else
        [sD, ind] = sort(mD, 2,'descend');
        for k=1:length(emptyClusters)
            M(:,emptyClusters(k)) = X(:,ind(k));
        end
    end
    dist_old = dist_new;
end
end

