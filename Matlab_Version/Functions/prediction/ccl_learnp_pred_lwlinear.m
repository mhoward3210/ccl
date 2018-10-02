function Yp = ccl_learnp_pred_lwlinear(X,model)
% Yp = ccl_learnp_pred_lwlinear(X,model)
%
% Locally weighted model prediction
%
% Input:
%
%   X                               State inputs
%   model                           Model parameters
%
% Ouput:
%
%   Yp                              Prediction




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

N  = size(X,2);       % get no. data points

% find feature vectors
Phi = model.phi(X);
dimPhi = size(Phi,1);

% find weights
W = model.W(X);
Nc = size(W,1);   % get no. data points, dimensionality

% predict training data
for nc=1:Nc
Yp(:,:,nc)=((repmat(W(nc,:),dimPhi,1).*Phi)'*model.w(:,:,nc))';
%Yp(:,:,nc) = sum(repmat(model.w(:,:,nc),1,N).*Phi).*W(nc,:);
end
Yp = sum(Yp,3)./repmat(sum(W,1),size(Yp,1),1);

