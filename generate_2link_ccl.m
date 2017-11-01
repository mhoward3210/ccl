function data = generate_2link_ccl (settings)
    
    Iu  = eye(settings.dim_u) ;   
    L   = settings.link.length ;
    nhat= settings.lambda ;
    
    % null space policy
    policy_ns = @(q) settings.null.alpha .* (settings.null.target - q) ;
    
    X = cell(settings.dim_k,1);
    F = cell(settings.dim_k,1);
    U = cell(settings.dim_k,1);  
    R = cell(settings.dim_k,1);
    B = cell(settings.dim_k,1);
    TS = cell(settings.dim_k,1);
    NS = cell(settings.dim_k,1);
    
    for k = 1: settings.nTraj

        % initial state: q1∼U[0◦, 10◦], q2∼U[90◦,100◦], q3∼U[0◦,10◦]
        x = (pi/180)*[0 90]' + (pi/180)*[10 10]'.*(rand(2,1));     
     
        X{k}        = zeros(settings.dim_u, settings.dim_n);
        F{k}        = zeros(settings.dim_u, settings.dim_n);
        U{k}        = zeros(settings.dim_u, settings.dim_n);
        R{k}        = zeros(settings.dim_r, settings.dim_n);
        B{k}        = zeros(settings.dim_r, settings.dim_n);
        TS{k}       = zeros(settings.dim_u, settings.dim_n);
        NS{k}       = zeros(settings.dim_u, settings.dim_n);
        
        for n = 1 : settings.dim_n+1                
            J   = jacobian (x, L) ;
            A   = nhat * J ;
            invA= pinv(A) ;
            P   = Iu - invA*A;
            f   = policy_ns(x);  
            ns  = P*f;
            r   = forward(x, L) ;             
            ts  = zeros(2,1) ;
            u   = ns ; 
     
            X{k}(:,n)   = x ;
            F{k}(:,n)   = f ;
            U{k}(:,n)   = u ;                
            R{k}(:,n)   = r ;          
            TS{k}(:,n)  = ts ;          
            NS{k}(:,n)  = ns ;          
            x           = x + u * settings.dt ;
           
            if ts'*ns > 1e-6            
                fprintf('ts and ns are not orthogonal') ;
                return ;
            elseif norm(ns) < 1e-3 %norm(ts) < 1e-3 || norm(ns) < 1e-3
                break ;
            end
        end % end n loop
        
        B{k} = diff(R{k}')' ;
        X{k} = X{k}(:,1:n-1) ;            
        R{k} = R{k}(:,1:n-1) ; 
        F{k} = F{k}(:,1:n-1) ;
        U{k} = U{k}(:,1:n-1) ;
        TS{k}= TS{k}(:,1:n-1) ;
        NS{k}= NS{k}(:,1:n-1) ;
        B{k} = B{k}(:,1:n-1) ;

        if settings.output.show_traj
        	visualise_move (R{k}, X{k},  settings.task.target, L, settings.task.A_index) ;
        end
       
    end % end k loop
            
    data.X = [X{:}];
    data.U = [U{:}];
    data.F = [F{:}];
    data.R = [R{:}]; 
    data.TS = [TS{:}]; 
    data.NS = [NS{:}]; 
end

function r = forward (q, L)
    r    = zeros(2,1) ;
    r(1) = L(1)*cos(q(1,:)) + L(2)*cos(q(1,:)+q(2,:)) ;
    r(2) = L(1)*sin(q(1,:)) + L(2)*sin(q(1,:)+q(2,:)) ;   
end

function J = jacobian (q, L)
    J(1,1) = -L(1)*sin(q(1,:)) - L(2)*sin(q(1,:)+q(2,:)) ;
    J(1,2) =                   - L(2)*sin(q(1,:)+q(2,:)) ;
    J(2,1) =  L(1)*cos(q(1,:)) + L(2)*cos(q(1,:)+q(2,:)) ;
    J(2,2) =                     L(2)*cos(q(1,:)+q(2,:)) ;    
end

function visualise_move (R, X, L)

    dim_n = length(R(1,:)) ;
    xmin = -1.5 ; xmax = 2 ;    
    ymin = -0.5 ; ymax = 2.5 ;
    
    fid_plot = 10 ;    
    fig_handle = figure(fid_plot); clf, hold on
        axis equal
        set(fig_handle, 'Position', [800, 100, 500, 500]);
        xlim([xmin,xmax]) ; ylim([ymin,ymax]) ; 
        plot( [R(1,1),R(1,end)], [R(2,1), R(2,end)], 'LineWidth', 5 ) ;
        % stroboscopic plot of arm        
        for i=1:round(dim_n/25):dim_n
            c = ((dim_n-i)/dim_n) * ones(1,3) ;
            plot_arm (X(:,i), L, c) ;
            pause (0.1) ;
        end
    hold off
end

function plot_arm (Q, L, C)    
    if ~exist('C', 'var')
        C = 'r' ;
    end    
    r1 = zeros(2,1); % base
   
    r2 = [L(1)*cos(Q(1));
          L(1)*sin(Q(1))];
    r3 = [r2(1) + L(2)*cos(Q(1)+Q(2));
          r2(2) + L(2)*sin(Q(1)+Q(2))];       
    plot([r1(1) r2(1)], [r1(2) r2(2)], 'LineStyle', '-', 'Color', C ) ;
    plot([r2(1) r3(1)], [r2(2) r3(2)], 'LineStyle', '-', 'Color', C ) ;
end