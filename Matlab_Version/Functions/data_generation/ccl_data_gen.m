function data = ccl_data_gen (settings)
% data = ccl_data_gen (settings)
%
% Generate a two link arm robot for simulation purpose by changing
% setting parameters for each constraint.
%
% Input:
%
%   settings:
%       settings.dim_u            Dimensionality of the action space
%       settings.dim_n            Length of the trajectories
%       settings.dim_k            Dimensionality of the constraint
%       settings.dim_r            Dimensionality of the task space x, z, orientation
%       settings.dt               Time interval
%       settings.nTraj            Dimensionality of the trajectories
%       settings.link.length      Length of robot arm links
%       settings.lambda           Ground truth of the selection matrix
%       settings.null.alpha       Null space  policy scaling
%       settings.null.target      Null space  policy target
%       settings.output.show_traj Use 1 to display generated data
%
% Output:
%
%   data:
%       data.X                    State space data
%       data.U                    Action space data
%       data.F                    Null space policy function handle
%       data.R                    Task space data
%       data.Ts                   Task space components data
%       data.Ns                   Null space components data

Iu  = eye(settings.dim_u) ;

% null space policy
policy_ns = settings.f_n;
% policy_ns = @(q) settings.null.alpha .* (settings.null.target - q) ;
% policy_ns = @(q) settings.null.alpha .* (q-1.*pi/180) ;

X = cell(settings.dim_k,1);
F = cell(settings.dim_k,1);
U = cell(settings.dim_k,1);
R = cell(settings.dim_k,1);
B = cell(settings.dim_k,1);
TS = cell(settings.dim_k,1);
NS = cell(settings.dim_k,1);
data.P = [];
if settings.grid_on == 1
    N = settings.N;
    xmax = ones(settings.dim_x,1); xmin=-xmax;                                % range of data
    Xtr  = repmat(xmax-xmin,1,N).*rand(settings.dim_x,N)+repmat(xmin,1,N);
    Ftr  = settings.f_n(Xtr);
    Btr  = settings.f_b(N);
    f_A  = settings.f_A;
    for n=1:N
        Atr(:,:,n)  = f_A(Xtr(:,n)) ;
        P   = eye(2) - pinv(Atr(:,:,n))*Atr(:,:,n) ;
        Ptr(:,:,n)  = P ;
        NStr(:,n)   = Ptr(:,:,n)*Ftr(:,n) ;
        TStr(:,n)   = pinv(Atr(:,:,n))*Btr(:,n) ;
        Ytr(:,n)    = TStr(:,n) + NStr(:,n) + settings.s2y*randn(settings.dim_u,1);
    end
    data.X = Xtr; data.Y = Ytr; data.N = N;
    data.F = Ftr; data.A = Atr; data.P = Ptr;
    data.NS = NStr ; data.TS = TStr ;
else

    for k = 1: settings.nTraj
        if strcmp(settings.task_policy_type,' ')
            policy_ts = @(x)zeros(settings.dim_k,1);
        else
            % task space policy
            target = settings.task.target(k);
            policy_ts = @(x) settings.null.alpha .* (target - x) ;
        end
        X{k}        = zeros(settings.dim_u, settings.dim_n);
        F{k}        = zeros(settings.dim_u, settings.dim_n);
        U{k}        = zeros(settings.dim_u, settings.dim_n);
        R{k}        = zeros(settings.dim_r, settings.dim_n);
        B{k}        = zeros(settings.dim_r, settings.dim_n);
        TS{k}       = zeros(settings.dim_u, settings.dim_n);
        NS{k}       = zeros(settings.dim_u, settings.dim_n);
        rnd = [1,-1];
        if isfield(settings,'learn_alpha')
            if strcmp(settings.null_policy_type,'linear_attractor')
                % initial state:
                x = ([1,2]'+rand(2,1)*0.1).*[rnd(randi(2)),rnd(randi(2))]';
            elseif strcmp(settings.null_policy_type,'limit_cycle')
                % initial state:
                x = ([0,0]'+rand(2,1)*0.1).*[rnd(randi(2)),rnd(randi(2))]';
            elseif strcmp(settings.null_policy_type,'linear')
                % initial state:
                x = ([1,2]'+rand(2,1)*0.1).*[rnd(randi(2)),rnd(randi(2))]';
            end
            
            if strcmp(settings.projection, 'state_independant')
                generate_A    = settings.f_alpha;                       % random constraint
            elseif strcmp(settings.projection, 'state_dependant')
                a = 2;
                f_alpha = @(x)([2*a*x(1),-1]);
                generate_A = @(x)(f_alpha(x));
            end
        end
        
        if isfield(settings,'learn_lambda')
            L   = settings.link.length ;
            x = (pi/180)*[0 90]' + (pi/180)*[10 10]'.*(rand(2,1));
            
            if strcmp(settings.projection, 'state_independant')
                generate_A = @(q)(settings.lambda * ccl_rob_jacobian (q, L));
            elseif strcmp(settings.projection, 'state_dependant')
                generate_A = @(q)(settings.lambda(q) * ccl_rob_jacobian (q, L));
            end
        end
        
        if settings.learn_nc == 1
            L   = settings.link.length ;
            x = (pi/180)*[0 90]' + (pi/180)*[10 10]'.*(rand(2,1));
            f_lambda = [0,1];                       % random constraint
            generate_A = @(q)(f_lambda * ccl_rob_jacobian (q, L));
        end
        
        if isfield(settings,'learn_pi')
            L   = settings.link.length ;
            x = (pi/180)*[0 90]' + (pi/180)*[10 10]'.*(rand(2,1));
            generate_A = settings.A;
        end

        
        for n = 1 : settings.dim_n+1
            A   = generate_A(x) ;
            invA= pinv(A) ;
            P_   = Iu - invA*A;
            f   = policy_ns(x);
            ns  = P_*f;
            if strcmp(settings.control_space,'joint')
                r   = ccl_rob_forward(x, L) ;
                ts  = pinv(A)*policy_ts(r(settings.fix_joint)) ;
                u   = ts + ns ;
                R{k}(:,n)   = r ;
                B{k} = diff(R{k}')' ;
                B{k} = B{k}(:,1:n-1) ;
            end
            if strcmp(settings.control_space,'end_effector')
                ts  = policy_ts(x(1)) ;
                u   = ns ;
            end
            X{k}(:,n)   = x ;
            F{k}(:,n)   = f ;
            U{k}(:,n)   = u ;
            TS{k}(:,n)  = ts ;
            NS{k}(:,n)  = ns ;
            P{k}(:,:,n) = P_;
            x           = x + u * settings.dt +settings.s2y*randn(settings.dim_u,1);
            
            if ts'*ns > 1e-6
                fprintf('ts and ns are not orthogonal') ;
                return ;
            elseif norm(ns) < 1e-3 %norm(ts) < 1e-3 || norm(ns) < 1e-3
                break ;
            end
        end % end n loop
        
        X{k} = X{k}(:,1:n-1) ;
        R{k} = R{k}(:,1:n-1) ;
        F{k} = F{k}(:,1:n-1) ;
        U{k} = U{k}(:,1:n-1) ;
        TS{k}= TS{k}(:,1:n-1) ;
        NS{k}= NS{k}(:,1:n-1) ;
        P{k} = P{k}(:,:,1:n-1) ;
        
        data.P = cat(3,data.P,P{k});
    end % end k loop
    if  strcmp(settings.control_space,'joint')
        if settings.output.show_traj
            ccl_rob_vis_move (R{end}, X{end}, L) ;
        end
    end
    data.X = [X{:}];
    data.U = [U{:}];
    data.F = [F{:}];
    data.R = [R{:}];
    data.TS = [TS{:}];
    data.NS = [NS{:}];
end