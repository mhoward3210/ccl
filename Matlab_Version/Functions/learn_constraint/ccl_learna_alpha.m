function [optimal] = ccl_learna_alpha (Un, X, options)
% [optimal] = ccl_learna_alpha (Un, X, options)
%
% Learning state dependent projection matrix N(q) for problem with the form
% Un = N(q) * F(q) where N is a state dependent projection matrix
%                        F is some policy
%
% Input:
%
%   X                                State of the system
%   Un                               Control of the system generated with the form Un(q) = N(q) * F(q)
%                                    where N(q)=I-pinv(A(q))'A(q) is the projection matrix that projects
%                                    F(q) unto the nullspace of A(q). N(q) can be state dependent, but
%                                    it should be generated in a consistent way.
%
% Output:
%   optimal                          A model for the projection matrix
%   optimal.f_proj(q)                A function that predicts N(q) given q




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

% essential parameters
model.dim_b     = options.dim_b ;   % dimensionality of the gaussian kernel basis
model.dim_r     = options.dim_r ;   % dimensionality of the end effector
model.dim_x     = size(X, 1) ;      % dimensionality of input
model.dim_u     = size(Un,1) ;      % dimensionality of output Un = N(X) * F(X) where X
model.dim_t     = model.dim_u - 1 ; % dimensionality of each constraint parameters
dim_n           = size(X,2) ;       % number of training points

% choose a method for generating the centres for gaussian kernel. A
% grid centre is usually adequate for a 2D problem. For higher
% dimensionality, kmeans centre normally performs better
if model.dim_x < 3
    model.dim_b = floor(sqrt(model.dim_b))^2 ;
    centres     = ccl_math_gc (X, model.dim_b) ;          % generate centres based on grid
else
    centres     = ccl_math_kc (X, model.dim_b) ;        % generate centres based on K-means
end
variance        = mean(mean(sqrt(ccl_math_distances(centres, centres))))^2 ; % set the variance as the mean distance between centres
model.phi       = @(x) ccl_basis_rbf ( x, centres, variance );   % gaussian kernel basis function
BX              = model.phi(X) ;                                    % K(X)

optimal.nmse    = 10000000 ;        % initialise the first model
model.var       = sum(var(Un,0,2)) ;% variance of Un

% The constraint matrix consists of K mutually orthogonal constraint vectors.
% At the k^{th} each iteration, candidate constraint vectors are
% rotated to the space orthogonal to the all ( i < k ) constraint
% vectors. At the first iteration, Rn = identity matrix
Rn = cell(1,dim_n) ;
for n = 1 : dim_n
    Rn{n} = eye(model.dim_u) ;
end

% The objective functions is E(Xn) = A(Xn) * Rn(Xn) * Un. For faster
% computation, RnUn(Xn) = Rn(Xn)*Un(Xn) is pre-caldulated to avoid
% repeated calculation during non-linear optimisation. At the first iteration, the rotation matrix is the identity matrix, so RnUn = Un
RnUn    = Un ;

Iu      = eye(model.dim_u) ;
for alpha_id = 1:model.dim_r
    model.dim_k = alpha_id ;
    model       = ccl_learna_sa (BX, RnUn, model ) ;                                 % search the optimal k^(th) constraint vector
    theta       = [pi/2*ones(dim_n, (alpha_id-1)), (model.w{alpha_id}* BX)' ] ;     % predict constraint parameters
    for n = 1: dim_n
        Rn{n}       = ccl_math_rotmat (theta(n,:), Rn{n}, model, alpha_id) ;    % update rotation matrix for the next iteration
        RnUn(:,n)   = Rn{n} * Un(:,n) ;                                             % rotate Un ahead for faster computation
    end
    % if the k^(th) constraint vector reduce the fitness, then the
    % previously found vectors are enough to describe the constraint
    if (model.nmse > optimal.nmse) && (model.nmse > 1e-3)
        break ;
    else
        optimal   = model ;
    end
end
optimal.f_proj  =  @(q) ccl_learna_pred_proj_alpha (optimal, q, Iu) ;  % a function to predict the projection matrix
fprintf('\t Found %d constraint vectors with residual error = %4.2e\n', optimal.dim_k, optimal.nmse) ;
end