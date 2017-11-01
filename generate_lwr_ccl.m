function dataset = generate_lwr_ccl (settings)
  
    settings.dim_hand       = 6 ;
    settings.dim_joint      = 7 ;               
    settings.joint.free     = setdiff(1:settings.dim_joint, settings.joint.fixed);         
    settings.joint.target   = pi/180 * zeros(7,1) ; % [10, -10, 10, -10, 10, -10, 10]' ;   
    settings.joint.limit    = pi/180 * [170, 120, 170, 120, 170, 120, 170]';        
    
    settings.end.free       = setdiff(1:settings.dim_hand, settings.end.fixed) ;
        
    settings.dim_u          = size(settings.joint.free,2) ;
    settings.dim_x          = size(settings.joint.free,2) ;    
    settings.dim_r          = size(settings.end.free,2) ;   
    
    %% set up the nullspace policy    
    switch (settings.null.type)
        case 'linear'      
            settings.null.alpha   = 1 ;
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
            r   = tr2diff(r) ;
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
    q = settings.joint.target ;
    q(settings.joint.free) = x ;
    Jxn = jacob0(rob,q) ;
    Jxn = Jxn (settings.end.free, settings.joint.free) ;
end

function ROBOT=dlr_7dof()
    L{1} = link([ pi/2 0 0 0 0 0],   'standard');
    L{2} = link([-pi/2 0 0 0.3  0 0],'standard');
    L{3} = link([-pi/2 0 0 0.4  0 0],'standard');
    L{4} = link([ pi/2 0 0 0.5  0 0],'standard');
    L{5} = link([ pi/2 0 0 0.39 0 0],'standard');
    L{6} = link([-pi/2 0 0 0    0 0],'standard');
    L{7} = link([ 0    0 0 0.2  0 0],'standard');
    ROBOT = robot(L, 'DLW III', 'DLR/KUKA', '');
end      