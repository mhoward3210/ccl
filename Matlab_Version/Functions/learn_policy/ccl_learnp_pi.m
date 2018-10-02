function model = ccl_learnp_pi(X,Y,model)
% model = ccl_learnp_pi(X,Y,model)
%
% Learn null space policy using regularised Least square method (parametric model)
%
% Input:
%
%   X                               Input data
%   Y                               Target data
%   model                           Parametric model parameters
%
% Output:
%
%   model                           learnt model parameters




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

[dimY N] = size(Y);

% find normalised Y
r = sum(Y.^2,1).^0.5;
YN = Y./repmat(r,dimY,1);

Phi   = model.phi(X);
dimPhi = size(Phi(:,1),1);

% construct Jacobian
YPhit = Y*Phi';
g = YPhit(:);

% construct Hessian
H = zeros(dimY*dimPhi);
for n=1:N
YNPhit = YN(:,n)*Phi(:,n)';
v(:,n) = YNPhit(:);
H = H + v(:,n)*v(:,n)';
end

% do eigendecomposition for inversion
%[V,D] = eig(H+1e-6*eye(size(H)));
[V,D] = eig(H);
ev = diag(D);
ind = find(ev>1e-6);
V1=V(:,ind);
pinvH1 = V1*diag(ev(ind).^-1)*V1';
model.w=reshape(pinvH1*g,dimY,dimPhi)';

