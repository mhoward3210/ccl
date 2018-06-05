function dataset = ccl_data_genlwr (settings)
% dataset = ccl_data_genlwr (settings)
%
% Generate Constraint Consistent Learning (CCL) data for a 7 Dof simulated robot arm
%
% Input:
%
%   settings                    Task related parameters (details see comments)
%
% Output:
%
%   dataset                     Generated dataset




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

settings.dim_hand       = 6 ;                                                   %Dimensionality of the task space
settings.dim_joint      = 7 ;                                                   %Dimensionality of the robot joints
settings.joint.free     = setdiff(1:settings.dim_joint, settings.joint.fixed);  %Labels of the free joints
settings.joint.target   = pi/180 * zeros(7,1) ;                                 %Target of the linear policy in joint space
settings.joint.limit    = pi/180 * [170, 120, 170, 120, 170, 120, 170]';        %Limits of each jiont

settings.end.free       = setdiff(1:settings.dim_hand, settings.end.fixed) ;    %Lables of the free dimensionality of the end effector/task space.

settings.dim_u          = size(settings.joint.free,2) ;                         %Dimensionality of the control commands
settings.dim_x          = size(settings.joint.free,2) ;                         %Dimensionality of the joint space
settings.dim_r          = size(settings.end.free,2) ;                           %Dmiensionaliry of the task space

%% set up the nullspace policy
switch (settings.null.type)                                                     %Null space policy types
    case 'linear'
        settings.null.alpha   = 1 ;                                             %Null space policy scaling
        settings.null.target  = settings.joint.target(settings.joint.free) ;
        policy_ns             = @(x) policy_linear(x, settings.null) ;
    case 'avoidance'
        settings.null.alpha   = 1 ;
        settings.null.target  = settings.joint.target(settings.joint.free) ;
        policy_ns             = @(x) policy_avoidance ( x, settings.null ) ;
    case 'learnt'
        policy_ns             = settings.null.func ;
    otherwise
        fprintf('Unkown null-space policy\n') ;
end

%% set-up the selection matrix for constraints
Lambda = settings.Lambda ;

rob = dlr_7dof ;

J   = @(q) jacob0(rob,q) ;
Jx  = @(x) get_jacobian (x, rob, settings) ;
dt  = settings.dt;
Iu  = eye(settings.dim_u) ;

X = cell(settings.dim_traj,1) ;
Pi= cell(settings.dim_traj,1) ;
U = cell(settings.dim_traj,1) ;
R = cell(settings.dim_traj,1) ;
V = cell(settings.dim_traj,1) ;

for k=1: settings.dim_traj
    
    %% get initial posture
    q = settings.joint.target ;
    q(settings.joint.free) = settings.joint.limit(settings.joint.free).*rand(settings.dim_u,1)- (settings.joint.limit(settings.joint.free)/2) ;
    x = q(settings.joint.free);
    
    X{k}    = zeros(settings.dim_u, settings.dim_step);
    U{k}    = zeros(settings.dim_u, settings.dim_step);
    Pi{k}   = zeros(settings.dim_u, settings.dim_step);
    
    for n = 1 : settings.dim_step+1
        q(settings.joint.free) = x ;
        Jn  = J(q) ;
        A   = Lambda * Jn(settings.end.free,settings.joint.free) ;
        invA= pinv(A) ;
        
        P   = Iu - invA*A ;
        f   = policy_ns(x) ;
        u   = P * f ;
        
        r   = fkine(rob,q) ;
        %             r   = tr2diff(r) ;
        r   = r(settings.end.free) ;
        
        X{k}(:,n)   = x ;
        U{k}(:,n)   = u ;
        R{k}(:,n)   = r ;
        Pi{k}(:,n)  = f ;
        x           = x + dt*u;
        
        if norm(u) < 1e-3
            break ;
        end
    end % end t loop
    V{k} = diff(R{k}')' ;
    V{k} = V{k}(:,1:n-1) ;
    X{k} = X{k}(:,1:n-1);
    R{k} = R{k}(:,1:n-1);
    U{k} = U{k}(:,1:n-1);
    Pi{k}= Pi{k}(:,1:n-1);
end % end k loop

dataset.X = [X{:}] ;
dataset.U = [U{:}] ;
dataset.Pi= [Pi{:}];
dataset.R = [R{:}] ;
dataset.V = [V{:}] ;
dataset.rob=rob ;
dataset.J = Jx ;
dataset.Lambda= Lambda ;
dataset.settings = settings ;
dataset.settings.dim_n = size(X,2) ;
end

function Jxn = get_jacobian (x, rob, settings)
% Jxn = get_jacobian (x, rob, settings)
%
% Return a function handle of jacobian calculation
%
% Input:
%   x                   Joint state variables
%   rob                 Robot object
%   settings            Task related parameters
%
% Output:
%   Jxn                 Jacobian function handle

q = settings.joint.target ;
q(settings.joint.free) = x ;
Jxn = jacob0(rob,q) ;
Jxn = Jxn (settings.end.free, settings.joint.free) ;
end

function ROBOT=dlr_7dof()
% ROBOT=dlr_7dof()
%
% Create a 7 Dof robot arm
%
% Output:
%   ROBOT                Robot object

L1 = Link([ 0 0 0 pi/2],   'standard');
L2 = Link([0 0 0.3 -pi/2],'standard');
L3 = Link([0 0 0.4 -pi/2],'standard');
L4 = Link([0 0 0.5 pi/2],'standard');
L5 = Link([0 0 0.39 pi/2],'standard');
L6 = Link([0 0 0 -pi/2],'standard');
L7 = Link([0 0 0.2 0],'standard');
ROBOT = SerialLink([L1,L2,L3,L4,L5,L6,L7], 'name', 'DLR/KUKA');
end
function U = policy_linear(x, null_settings)
% U = policy_linear(x, null_settings)
%
% A linear null space policy
%
% Input:
%   x                   Joint state variable
%   null_settings       Setting parameters for null space policy
% Output:
%   U                   Null space policy function handle
U = null_settings.alpha .* ((null_settings.target - x));
end