function SbS_vis (settings,f_project)
% SbS_vis (settings)
%
% Side by side visualisation of the learnt constraint
%
% Input:
%
%   settings                       Parameters
%   f_project                      Learnt constraint

Iu  = eye(settings.dim_u) ;
L   = settings.link.length ;

% null space policy
policy_ns = @(q) settings.null.alpha .* (settings.null.target - q) ;
% policy_ns = @(q) settings.null.alpha .* (q-1.*pi/180) ;

x = (pi/180)*[0 90]' + (pi/180)*[10 10]'.*(rand(2,1));
x1 = x;
if strcmp(settings.projection,'state_independant')
    nhat = @(x)(settings.lambda) ;
elseif strcmp(settings.projection,'state_dependant')
    a = 2;
    nhat = @(x)([2*a.*[1,0]*forward(x,settings.link.length), -1]) ;
end

for n = 1 : settings.dim_n+1
    J   = jacobian (x, L) ;
    r   = forward(x, L) ;       r_   = forward(x1, L) ;
    A   = nhat(x) * J ;
    invA= pinv(A) ;
    P_  = Iu - invA*A;
    f   = policy_ns(x);         f_   = policy_ns(x1);
    ns  = P_*f;                 ns_  = real(f_project(x1))*f_;
    ts  = zeros(2,1) ;
    u   = ns ;                  u_   = ns_ ;
    x   = x + u * settings.dt;
    x1  = x1 + u_ * settings.dt;
    
    if ts'*ns > 1e-6
        fprintf('ts and ns are not orthogonal') ;
        return ;
    elseif norm(ns) < 1e-3 %norm(ts) < 1e-3 || norm(ns) < 1e-3
        break ;
    end
    R(:,n) = r;
    R_(:,n) = r_;
    X(:,n) = x;
    X_(:,n) = x1;
end % end n loop
visualise_move (R, X, L,'Movement under true constraint',[500,200,500,500]) ;
visualise_move (R_, X_, L,'Movement under Learnt constraint',[1000,200,500,500]) ;
end