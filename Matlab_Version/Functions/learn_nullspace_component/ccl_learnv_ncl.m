function model = ccl_learnv_ncl(X, Y, model)
% model = ccl_learnv_ncl(X, Y, model)
%
% Learn the nullspace component of the input data X

% Input:
%
%   X                                Observed states
%   Y                                Observed actions of the form Y = A(X)'B(X) + N(X) F(X) where N and F
%                                    are consistent across the dataset X
%
% Output:
%
%   model                           Model that prodicts the nullspace component N(X)F(X)




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

% calculate the basis functions of the training data
BX      = model.phi(X) ;

% learn an initial model using a simple parametric model
model   = learn_model_dir ( model, BX, Y ) ;

% learn the model by minimising the proposed step-1 method
obj     = @(W) ccl_obj_ncl ( model, W, BX, Y);   % setup the learning objective function
options = optimset( 'Jacobian','on', 'Display', 'notify',...  % options for the optimisation
    'MaxFunEvals',1e9, 'MaxIter', 1000,...
    'TolFun',1e-9, 'TolX',1e-9,...
    'Algorithm', 'levenberg-marquardt');
model.w = lsqnonlin(obj, model.w, [], [], options );  % use the non-linear optimiser to solve obj_ncl_reg
end

function model = learn_model_dir ( model, BX, U )
% model = learn_model_dir ( model, BX, U )
%
% Direct policy learning approach
%
% Input:
%   model                           Model related parameters
%   BX                              High demensionality of the input data
%
% Output:
%   model                           Learnt model

HS  = eye(model.num_basis);
g   = BX * U' ;
H   = BX * BX';

% do eigen-decomposition for inversion
[V,D]   = eig( H + 1e-8*HS);
ev      = diag(D);
ind     = find( ev > 1e-8);
V1      = V(:,ind);
pinvH1  = V1 * diag(ev(ind).^-1)*V1';
model.w = (pinvH1 * g)' ;
end
