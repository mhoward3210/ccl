clear all ; close all ;      
rand('seed',1),randn('seed',1)

settings.dim_n          = 50 ;  % number of steps in each trajactory            
settings.nTraj          = 20 ;  % number of trajectories        
settings.dim_exp        = 5 ;   % number of experiment to repeat

settings.dim_x          = 2 ;   % dimensionality of the state space
settings.dim_u          = 2 ;   % dimensionality of the action space
settings.dim_r          = 2 ;   % dimensionality of the task space x, z, orientation
settings.dim_k          = 1 ;   % dimensionality of the constraint
settings.dim_b          = 16 ;  % dimensionality of the kernel function. 16 is normally enough for a 2D problem  
settings.dt             = 0.05 ;

settings.null.alpha     = 1 ;                   % null space policy scaling
settings.null.target    = pi/180*[10, -10]' ;   % null space target
settings.link.length    = [1, 1] ;              % length of the robot

settings.output.show_traj= 0 ;                  % use 1 to display generated data

for i = 1:settings.dim_exp
    
    % set up a random constraint
    lambda = rand(settings.dim_k, settings.dim_u) ;
    settings.lambda = lambda ./ norm(lambda) ;

    %% generate training and testing data 
    Dtr = generate_2link_ccl (settings) ;
    Dte = generate_2link_ccl (settings) ;
    Xtr = Dtr.X ; Ytr = Dtr.U ; TStr = Dtr.TS ; NStr = Dtr.NS ; Ftr = Dtr.F ; 
    Xte = Dte.X ; Yte = Dte.U ; TSte = Dtr.TS ; NSte = Dte.NS ; Fte = Dte.F ;          

    fprintf(1,'\n Experiment %d \n', i) ;
    fprintf(1,'\t Dimensionality of action space: %d \n', settings.dim_u) ;
    fprintf(1,'\t Dimensionality of task space:   %d \n', settings.dim_k) ;
    fprintf(1,'\t Dimensionality of null space:   %d \n', settings.dim_u-settings.dim_k) ;
    fprintf(1,'\t Dimensionality of kernel:       %d \n', settings.dim_b) ;
    fprintf(1,'\t Size of the training data:      %d \n', size(Xtr,2)) ; 
    fprintf(1,'\t Random constraint:  [%4.2f, %4.2f] \n', settings.lambda(1), settings.lambda(2)) ;
    fprintf(1,'\n Learning state-dependent constraint vectors... \n') ;
    
    %% learn alpha (constraint vector) for problem with unknown Jacobian       
    model_alpha_ccl  = learn_alpha_ccl (NStr, Xtr, settings);      
    ppe.alpha_ccl.tr = get_ppe_alpha (model_alpha_ccl.f_proj, Xtr, Ftr, NStr) ;
    ppe.alpha_ccl.te = get_ppe_alpha (model_alpha_ccl.f_proj, Xte, Fte, NSte) ;
    poe.alpha_ccl.tr = get_poe_alpha (model_alpha_ccl.f_proj, Xtr, NStr) ;        
    poe.alpha_ccl.te = get_poe_alpha (model_alpha_ccl.f_proj, Xte, NSte) ;   

    fprintf(1,'\n Result %d \n', i ) ;    
    fprintf(1,'\t ===============================\n' ) ;
    fprintf(1,'\t       |    NPPE   |   NPOE \n' ) ;
    fprintf(1,'\t -------------------------------\n' ) ;
    fprintf(1,'\t Train |  %4.2e | %4.2e  \n', ppe.alpha_ccl.tr,  poe.alpha_ccl.tr ) ;
    fprintf(1,'\t Test  |  %4.2e | %4.2e  \n', ppe.alpha_ccl.te,  poe.alpha_ccl.te ) ;
    fprintf(1,'\t ===============================\n' ) ;
end
