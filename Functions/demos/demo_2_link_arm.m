function demo_2_link_arm
% demo_2_link_arm
% This demo demonstrates the usage of CCL library with a simulated 2-Link
% robot arm which works in x,y plane of the operaitonal space.
clear all;clc;close all;rng('default');
fprintf('<=========================================================================>\r');
fprintf('<=========================================================================>\r');
fprintf('<===========   Constraint Consistent Learning Library    =================>\r');
fprintf('< This demo will go through a 2 link arm example to demonstrate this CCL  >\r');
fprintf('< toolbox. The CCL is formulated in the following format:                 > \r');
fprintf('< Consider the set of consistent k-dimensional constraints:               >\r');
fprintf('<                  A(x)U(x) = b(x)   where A(x) = lamba(x)J(x)            >\r');
fprintf('<                      U(x) = pinv(A(x))b(x) + (I-pinv(A(x))A(x))Pi(x)    >\r');
fprintf('<                      U(x) =      U_ts      +         U_ns               >\r');
fprintf('< The task is defined in 2D. The constraints are either random or state   >\r');
fprintf('< dependant parabola. The null space control policies are either linear   >\r');
fprintf('< attractor or limit cycle. This demo will execute section by section and >\r');
fprintf('< allow the user to configure the training parameters.                    >\r');
fprintf('< List of sections:                                                       >\r');
fprintf('< SECTION 1:       PARAMETER CONFIGURATION                                >\r');
fprintf('< SECTION 2:       LEARNING NULL SPACE COMPONENTS                         >\r');
fprintf('< SECTION 3:       LEARNING NULL SPACE CONSTRAINTS                        >\r');
fprintf('< SECTION 4:       LEARNING NULL SPACE POLICY                             >\r');
fprintf('< Configuration options:                                                  >\r');
fprintf('< Constraints:                                                            >\r');
fprintf('<            State independant:                                           >\r');
fprintf('<                              linear (random)                            >\r');
fprintf('<              State dependant:                                           >\r');
fprintf('<                              parabola                                   >\r');
fprintf('< Null space policy:                                                      >\r');
fprintf('<                  linear attractor                                       >\r');
fprintf('< Task space policy:                                                      >\r');
fprintf('<                  random                                                 >\n');
fprintf('<=========================================================================>\r');
fprintf('<=========================================================================>\r');
fprintf('<=========================================================================>\n\n\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('< SECTION 1:       PARAMETER CONFIGURATION                                >\r');
fprintf('\n< User specified configurations are:                                      >\r');
%% GENERATIVE MODEL PARAMETERS
settings.dim_x          = 2 ;                                   % dimensionality of the state space
settings.dim_u          = 2 ;                                   % dimensionality of the action space
settings.dim_r          = 2 ;                                   % dimensionality of the task space
settings.dim_k          = 1 ;                                   % dimensionality of the constraint
settings.dt             = 0.01 ;                                % time step
settings.s2y  = .01;                                            % noise in output
settings.null.alpha     = 1 ;                                   % null space policy scaling
settings.null.target    = pi/180*[10, -10]';                    % null space target
settings.task.target    = @(n)(randi([-2,2])*rand(1)) ;         % task space target
settings.link.length    = [1, 1] ;                              % length of the robot
settings.projection = 'state_independant';                        % {'state_independant' 'state_dependant'}
settings.task_policy_type = 'linear_attractor';                 % {'linear_attractor'}
settings.null_policy_type = 'linear_attractor';                 % {'linear_attractor'}
settings.f_n = @(q) settings.null.alpha .* (settings.null.target - q) ;

J = @(q)jacobian(q,settings.link.length);                       % robot jacobian
settings.output.show_traj = 0 ;                                  % use 1 to display generated data
settings.control_space    = 'joint';                     % control space in joint
fprintf('< Dim_x             = %d                                                   >\r',settings.dim_x);
fprintf('< Dim_u             = %d                                                   >\r',settings.dim_u);
fprintf('< Dim_r             = %d                                                   >\r',settings.dim_r);
fprintf('< Dim_k             = %d                                                   >\r',settings.dim_k);
fprintf('< Constraint        = %s                                                   >\r',settings.projection);
fprintf('< Null_policy_type  = %s                                                   >\r',settings.null_policy_type);
fprintf('< Task_policy_type  = %s                                                   >\r',settings.task_policy_type);
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
pause();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              LEARN NULL SPACE COMPONENT U_n = U_ts + U_ns                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n\n<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('< SECTION 2:       LEARNING NULL SPACE COMPONENTS                         >\r');
fprintf('< In this section,we are trying to learn U_ns out of the observations(X,U)>\r');
fprintf('< The cost function to be minimised is:                                   >\r');
fprintf('<              E[Pi_ns] = sum(||P*U - Pi_ns)||.^2                         >\r');
fprintf('< In order to achieve good performance, for each constraint, enough data  >\r');
fprintf('< samples and model complexity are necessary                              >\n');
fprintf('\n< For details please refer to:                                            >\r');
fprintf('< Towell, M. Howard, and S. Vijayakumar. IEEE International Conference    >\r');
fprintf('< Intelligent Robots and Systems, 2010.                                   >\n');
fprintf(1,'\n< Start learning Null space component  ...                                > \r');
settings.dim_exp        = 1 ;                                         % number of experiment to repeat
settings.dim_n          = 40 ;                                        % number of steps in each trajactory
settings.nTraj          = 40 ;                                        % number of trajectories
settings.grid_on        = 0  ;
settings.learn_nc       = 1  ;
model = [];
for i = 1:settings.dim_exp
%     fix = [0,1] ;
%     settings.lambda = lambda;
    settings.fix_joint = 1;
    model.num_basis = 100 ;                                           % define the number of radial basis functions
    
    fprintf(1,'\n Experiment %d \n', i) ;
    fprintf(1,'\t Dimensionality of kernel:       %d \n', model.num_basis) ;
    fprintf(1,'\t Number of trajectories:       %d \n', settings.nTraj) ;
    fprintf(1,'\t Number of time steps in each trajectory:       %d \n', settings.dim_n) ;

    fprintf('\n< Generating training dataset for learning null space components  ...     >\r');
    Dtr = generate_data_ccl (settings) ;
    Xtr = Dtr.X ; Ytr = Dtr.U ; TStr = Dtr.TS ; NStr = Dtr.NS ; Ftr = Dtr.F ;
    fprintf(1,'#Data (train): %5d, \r',settings.dim_n*settings.nTraj);
    fprintf('< Generating testing dataset for learning null space components   ...     >\n');
    Dte = generate_data_ccl (settings) ;
    Xte = Dte.X ; Yte = Dte.U ; TSte = Dtr.TS ; NSte = Dte.NS ; Fte = Dte.F ;
    fprintf(1,'#Data (test): %5d, \n',settings.dim_n*settings.nTraj);
    
    % set up the radial basis functions
    model.c     = generate_grid_centres (Xtr, model.num_basis) ;      % generate a grid of basis functions
    model.s2    = mean(mean(sqrt(distances(model.c, model.c))))^2 ;   % set the variance as the mean distance between centres
    model.phi   = @(x)phi_gaussian_rbf ( x, model.c, model.s2 );      % normalised Gaussian rbfs
    
    % learn the nullspace component
    model       = learn_ncl (Xtr, Ytr, model) ;                       % learn the model
    f_ncl       = @(x) predict_ncl ( model, x ) ;                     % set up an inference function
    
    % predict nullspace components
    NSptr = f_ncl (Xtr) ;
    NSpte = f_ncl (Xte) ;
    
    % calculate errors
    NUPEtr = get_nupe(NStr, NSptr) ;
    NUPEte = get_nupe(NSte, NSpte) ;
    YNSptr = get_npe (Ytr,  NSptr) ;
    YNSpte = get_npe (Yte,  NSpte) ;
    fprintf(1,'NUPE (train) = %5.3e, ', NUPEtr);
    fprintf(1,'NNPE (train) = %5.3e, ', YNSptr);
    fprintf(1,'\n');
    fprintf(1,'NUPE (test)  = %5.3e, ', NUPEte);
    fprintf(1,'NNPE (test)  = %5.3e, ', YNSpte);
    fprintf(1,'\n');
    fprintf('<=========================================================================>\n');
    fprintf('<=========================================================================>\n');
    fprintf('<=========================================================================>\n');
end
pause();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                            LEARN CONSTRAINTS                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n\n<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('< SECTION 3:       LEARNING NULL SPACE CONSTRAINTS                        >\r');
fprintf('< In this section,we will start addressing state independant constraint A >\r');
fprintf('< and state dependant constraint A(x). The exploration policy will change >\r');
fprintf('< change the performance of the learnt constraint                         >\r');
fprintf('< The cost function to be minimised is:                                   >\r');
fprintf('<                           E[A] = min(sum||A*Un||^2)                     >\n');
fprintf('\n< For details please refer to:                                            >\r');
fprintf('< H.-C. Lin, M. Howard, and S. Vijayakumar. IEEE International Conference >\r');
fprintf('< Robotics and Automation, 2015                                           >\n');
settings.dim_b          = 10 ;                          % dimensionality of the kernal
settings.dim_n          = 40 ;                          % number of steps in each trajactory
settings.nTraj          = 40 ;                          % number of trajectories
settings.dim_exp        = 1  ;                          % number of experiment to repeat
settings.task_policy_type = ' ';
settings.learn_nc       = 0  ;
settings.learn_lambda   = 1  ;
for i = 1:settings.dim_exp
    if strcmp(settings.projection, 'state_independant')
        settings.lambda = orth( rand(2, 1) )';      
        fprintf(1,'\n< Start learning state independant null space projection   ...            > \n');
        % set up a random constraint
        fprintf('\n< Generating training dataset for learning constraints    ...             >\r');
        Dtr = generate_data_ccl (settings) ;
        fprintf(1,'#Data (train): %5d, \r',settings.dim_n*settings.nTraj);
        fprintf('< Generating testing dataset for learning constraints     ...             >\r');
        Dte = generate_data_ccl (settings) ;
        fprintf(1,'#Data (test): %5d, \n',settings.dim_n*settings.nTraj);
        
    elseif strcmp(settings.projection, 'state_dependant')
        settings.lambda = @(q)return_lambda(q,settings.link.length);
        fprintf(1,'\n< Start learning state dependant null space projection   ...              > \n');
        fprintf('\n< Generating training dataset for learning constraints    ...             >\r');
        Dtr = generate_data_ccl (settings) ;
        fprintf(1,'#Data (train): %5d, \r',settings.dim_n*settings.nTraj);
        fprintf('< Generating testing dataset for learning constraints     ...             >\r');
        Dte = generate_data_ccl (settings) ;
        fprintf(1,'#Data (test): %5d, \n',settings.dim_n*settings.nTraj);
        
    end
    %% generate training and testing data
    Xtr = Dtr.X ; Ytr = Dtr.U ; TStr = Dtr.TS ; NStr = Dtr.NS ; Ftr = Dtr.F ;
    Xte = Dte.X ; Yte = Dte.U ; TSte = Dtr.TS ; NSte = Dte.NS ; Fte = Dte.F ;
    
    fprintf(1,'\n Experiment %d \n', i) ;
    fprintf(1,'\t Dimensionality of kernel:       %d \n', settings.dim_b) ;
    fprintf(1,'\t Number of trajectories:       %d \n', settings.nTraj) ;
    fprintf(1,'\t Number of time steps in each trajectory:       %d \n', settings.dim_n) ;
    
    %% learn alpha (constraint vector) for problem with unknown Jacobian
    fprintf(1,'\n Learning state-dependent constraint vectors withiout prior knowledge ... >\r') ;
    model_alpha_ccl  = learn_alpha_ccl (NStr, Xtr, settings);
    ppe.alpha_ccl.tr = get_ppe_alpha (model_alpha_ccl.f_proj, Xtr, Ftr, NStr) ;
    ppe.alpha_ccl.te = get_ppe_alpha (model_alpha_ccl.f_proj, Xte, Fte, NSte) ;
    poe.alpha_ccl.tr = get_poe_alpha (model_alpha_ccl.f_proj, Xtr, NStr) ;
    poe.alpha_ccl.te = get_poe_alpha (model_alpha_ccl.f_proj, Xte, NSte) ;
    
    fprintf(1,'\n Result learn alpha %d \n', i ) ;
    fprintf(1,'\t ===============================\n' ) ;
    fprintf(1,'\t       |    NPPE   |   NPOE \n' ) ;
    fprintf(1,'\t -------------------------------\n' ) ;
    fprintf(1,'\t Train |  %4.2e | %4.2e  \n', ppe.alpha_ccl.tr,  poe.alpha_ccl.tr ) ;
    fprintf(1,'\t Test  |  %4.2e | %4.2e  \n', ppe.alpha_ccl.te,  poe.alpha_ccl.te ) ;
    fprintf(1,'\t ===============================\n' ) ;
    
    %% learn alpha (constraint vector) for problem with known Jacobian
    fprintf(1,'\n Learning state-dependent constraint vectors with prior knowledge ...    >\r') ;
    model_lambda_ccl  = learn_lambda_ccl (NStr, Xtr, J, settings);
    ppe.lambda_ccl.tr = get_ppe_alpha (model_lambda_ccl.f_proj, Xtr, Ftr, NStr) ;
    ppe.lambda_ccl.te = get_ppe_alpha (model_lambda_ccl.f_proj, Xte, Fte, NSte) ;
    poe.lambda_ccl.tr = get_poe_alpha (model_lambda_ccl.f_proj, Xtr, NStr) ;
    poe.lambda_ccl.te = get_poe_alpha (model_lambda_ccl.f_proj, Xte, NSte) ;
    
    fprintf(1,'\n Result learn lambda %d \n', i ) ;
    fprintf(1,'\t ===============================\n' ) ;
    fprintf(1,'\t       |    NPPE   |   NPOE \n' ) ;
    fprintf(1,'\t -------------------------------\n' ) ;
    fprintf(1,'\t Train |  %4.2e | %4.2e  \n', ppe.lambda_ccl.tr,  poe.lambda_ccl.tr ) ;
    fprintf(1,'\t Test  |  %4.2e | %4.2e  \n', ppe.lambda_ccl.te,  poe.lambda_ccl.te ) ;
    fprintf(1,'\t ===============================\n' ) ;
    
    %% visualisation of the constraint
    visualise_sbs (settings,model_lambda_ccl.f_proj)
    
end
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
pause();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              LEARN PARAMETRIC NULL SPACE POLICY MODEL                              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Learn null space policy %%
fprintf('\n\n\n<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('< SECTION 4:       LEARNING NULL SPACE POLICY                             >\r');
fprintf('< This section is for learning null space policy models. The methods we   >\r');
fprintf('< use are parametric model and locally weighted parametric model          >\r');
fprintf('< The cost function to be minimised is:                                   >\r');
fprintf('<                  E[Pi] = min(sum||U_ns - U_ns^2*Pi||^2)                 >\r');
fprintf('< To achiev a better unconstrained null space policy, more constraints    >\r');
fprintf('< need to be demonstrated in the training dataset                         >\n');
fprintf('\n< For details please refer to:                                            >\r');
fprintf('< Howard, Matthew, et al. "A novel method for learning policies from      >\r');
fprintf('< variable constraint data." Autonomous Robots 27.2 (2009): 105-121.      >\n');
fprintf('\n< Start learning null space policy   ...                                 > \n');
Dtr     = [];
Dte     = [];
settings.dim_n          = 40 ;  % number of steps in each trajactory
settings.nTraj          = 40 ;  % number of trajectories
model.dim_b             = 25;   % dimensionality of the kernal
settings.dim_exp        = 1;
settings.learn_pi       = 1  ;
if strcmp(settings.projection, 'state_independant')
    settings.A = @(q)(orth( rand(2, 1) )' * jacobian (q, settings.link.length));
elseif strcmp(settings.projection, 'state_dependant')
    lambda = @(q)return_lambda(q,settings.link.length,2);
    settings.A = @(q)(lambda(q) * jacobian (q, settings.link.length));
end

for i = 1:settings.dim_exp
    fprintf('\n< Generating training dataset for learning null space constraints  ...    >\n');
    Dtr = generate_data_ccl (settings) ;
    fprintf(1,'#Data (train): %5d, \r',settings.dim_n*settings.nTraj);
    fprintf('\n< Generating testing dataset for learning null space constraints   ...    >\n');
    Dte = generate_data_ccl (settings) ;
    fprintf(1,'#Data (test): %5d, \n',settings.dim_n*settings.nTraj);
    Xtr = Dtr.X ; Ytr = Dtr.U ; TStr = Dtr.TS ; NStr = Dtr.NS ; Ftr = Dtr.F ; Ptr = Dtr.P;
    Xte = Dte.X ; Yte = Dte.U ; TSte = Dtr.TS ; NSte = Dte.NS ; Fte = Dte.F ; Pte = Dte.P;
    fprintf(1,'\n Experiment %d \n', i) ;
    fprintf(1,'\t Dimensionality of kernel:       %d \n', settings.dim_b) ;
    fprintf(1,'\t Number of trajectories:       %d \n', settings.nTraj) ;
    fprintf(1,'\t Number of time steps in each trajectory:       %d \n', settings.dim_n) ;
    
    % set up the radial basis functions
    model.c     = generate_grid_centres (Xtr, model.dim_b) ;      % generate a grid of basis functions
    model.s2    = mean(mean(sqrt(distances(model.c, model.c))))^2 ;   % set the variance as the mean distance between centres
    model.phi   = @(x)phi_gaussian_rbf ( x, model.c, model.s2 );      % normalised Gaussian rbfs
    
    % train parametric model
    model = learn_ccl_pi(Xtr,NStr,model); fp = @(x)predict_linear(x,model);
    
    % predict training data
    Fptr = fp(Xtr);
    
    % compute training error
    NUPEtr = get_nupe(Ftr,Fptr);
    fprintf(1,'NUPE (train) = %5.3e, ',NUPEtr);
    NCPEtr = get_ncpe(NStr,Fptr,Ptr);
    fprintf(1,'NCPE (train) = %5.3e, ',NCPEtr);
    
    % predict test data
    Fpte = fp(Xte);
    
    % compute test error
    NUPEte = get_nupe(Fte,Fpte);
    fprintf(1,'NUPE (test) = %5.3e, ',NUPEte);
    NCPEte = get_ncpe(NSte,Fpte,Pte);
    fprintf(1,'NCPE (test) = %5.3e',NCPEte);
    fprintf(1,'\n');
end
pause();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              LEARN LOCALLY WEIGHTED LINEAR POLICY MODEL                            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(1,'\n< Start learning Null space policy with locally weighted  model   ...     > \n');
model   = [];
Dtr     = [];
Dte     = [];
model.dim_b = 25;
settings.dim_exp = 1;

for i = 1:settings.dim_exp
    fprintf(1,'\n Experiment %d \n', i) ;
    fprintf(1,'\t Dimensionality of kernel:       %d \n', settings.dim_b) ;
    fprintf(1,'\t Number of trajectories:       %d \n', settings.nTraj) ;
    fprintf(1,'\t Number of time steps in each trajectory:       %d \n', settings.dim_n) ;
    % set up the radial basis functions
    model.c     = generate_grid_centres (Xtr, model.dim_b) ;          % generate a grid of basis functions
    model.s2    = mean(mean(sqrt(distances(model.c, model.c))))^2 ;   % set the variance as the mean distance between centres
    model.phi   = @(x)phi_gaussian_rbf ( x, model.c, model.s2 );      % normalised Gaussian rbfs
    
    model.W   = @(x)phi_gaussian_rbf( x, model.c, model.s2 );
    model.phi = @(x)phi_linear( x );
    %model.phi = @(x)phi_gaussian_rbf ( x, model.c, model.s2 );
    
    % train the model
    model = learn_lwccl(Xtr,NStr,model); fp = @(x)predict_local_linear(x,model);
    
    % predict training data
    Fptr = fp(Xtr);
    
    % compute training error
    NUPEtr = get_nupe(Ftr,Fptr);
    fprintf(1,'NUPE (train) = %5.3e, ',NUPEtr);
    NCPEtr = get_ncpe(NStr,Fptr,Ptr);
    fprintf(1,'NCPE (train) = %5.3e, ',NCPEtr);
    
    % predict test data
    Fpte = fp(Xte);
    
    % compute test error
    NUPEte = get_nupe(Fte,Fpte);
    fprintf(1,'NUPE (test) = %5.3e, ',NUPEte);
    NCPEte = get_ncpe(NSte,Fpte,Pte);
    fprintf(1,'NCPE (test) = %5.3e',NCPEte);
    fprintf(1,'\n');
end
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
pause();
close all;
end