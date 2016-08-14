clear all ; close all ;      
rand('seed',1),randn('seed',1)

settings.dim_x          = 7 ;   % dimensionality of the state space
settings.dim_u          = 7 ;   % dimensionality of the action space
settings.dim_r          = 3 ;   % dimensionality of the task space x, z, orientation
settings.dim_step       = 50 ;  % number of steps in each trajactory            
settings.dim_traj       = 50 ;  % number of trajectories        
settings.dt             = 0.05 ; 
settings.link.length    = [1, 1, 0.5] ;             % length of the robot

%% choose the constraints
constraint_list{1} = [0, 0, 1 ] ;
constraint_list{2} = [0, sin(pi/3), cos(pi/3) ] ;
constraint_list{3} = [1, 0, 0; 0, 1, 0 ] ;
constraint_list{4} = [0, sin(pi/3), cos(pi/3) ; sin(pi/2)*sin(pi/3), sin(pi/2)*cos(pi/3), cos(pi/2) ] ;
dim_constraint     = length(constraint_list) ;    

settings.null.alpha     = 1 ;                       % null space policy scaling
settings.null.type      = 'linear' ;

settings.joint.fixed    = [];      
settings.end.fixed      = [4,5,6] ;

%% parameters needed for learning the projection matrix
settings.search.dim_u       = settings.dim_u ;          % dimensionality of the joint space
settings.search.dim_r       = settings.dim_r ;          % dimensionality of the task space 

Dtr = cell(dim_constraint,1) ;
Dte = cell(dim_constraint,1) ;

for p_id = 1:dim_constraint 
    %% Generate training and testing data     
    settings.Lambda = constraint_list{p_id} ;
    Dtr{p_id} = generate_lwr_ccl (settings) ;  
    Dte{p_id} = generate_lwr_ccl (settings) ;  
    Xtr = Dtr{p_id}.X ; Ytr = Dtr{p_id}.U ; Ftr = Dtr{p_id}.Pi ; Vtr = Dtr{p_id}.V ;
    Xte = Dte{p_id}.X ; Yte = Dte{p_id}.U ; Fte = Dte{p_id}.Pi ; Vte = Dte{p_id}.V ;
       
    %% learning projection matrix    
    fprintf(1,'Constraint %d \n', p_id) ;
    model_lambda  = learn_lambda_ccl (Ytr, Xtr, Dtr{p_id}.J, settings.search) ;   
    
    %% calculate model error    
    ppe.lambda.tr = get_ppe_alpha (model_lambda.f_proj, Xtr, Ftr, Ytr) ;
    ppe.lambda.te = get_ppe_alpha (model_lambda.f_proj, Xte, Fte, Yte) ;
    poe.lambda.tr = get_poe_alpha (model_lambda.f_proj, Xtr, Ytr) ;           
    poe.lambda.te = get_poe_alpha (model_lambda.f_proj, Xte, Yte) ;      
    
    fprintf(1,'-------------------------------------\n' ) ;
    fprintf(1,' Dataset  |   NPPE   |   NPOE          \n' ) ;
    fprintf(1,'-------------------------------------\n' ) ;
    fprintf(1,' Training | %4.2e | %4.2e \n', ppe.lambda.tr, poe.lambda.tr ) ;
    fprintf(1,' Testing  | %4.2e | %4.2e \n', ppe.lambda.te, poe.lambda.te ) ;    
end