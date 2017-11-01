function demo_toy_example_2D
% demo_toy_example_2D
% This demo demonstrates a toy 2D problem for the usage of the CCL library.
clear all;clc;close all;rng('default');
fprintf('<=========================================================================>\r');
fprintf('<=========================================================================>\r');
fprintf('<===========   Constraint Consistent Learning Library    =================>\r');
fprintf('< This demo will go through a Toy example to demonstrate this CCL toolbox >\r');
fprintf('< The CCL is formulated in the following format:                          > \r');
fprintf('< Consider the set of consistent k-dimensional constraints:               >\r');
fprintf('<                  A(x)U(x) = b(x)                                        >\r');
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
fprintf('<                  linear                                                 >\r');
fprintf('<                  limit cycle                                            >\r');
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
ctr = .9*ones(1,3); lwtr = 1;                                   % colour, linewidth for training data
cv  = [0  0  0];    lwv  = 1;                                   % colour, linewidth for visualisation data
cpv = [1 .5 .5];    lwpv = 2;                                   % colour, linewidth for visualisation data predictions
settings.dim_x          = 2 ;                                   % dimensionality of the state space
settings.dim_u          = 2 ;                                   % dimensionality of the action space
settings.dim_r          = 2 ;                                   % dimensionality of the task space
settings.dim_k          = 1 ;                                   % dimensionality of the constraint
settings.dt             = 0.1;                                  % time step
settings.null.alpha     = 0.5 ;                                   % null space policy scaling
settings.s2y  = .01;                                                     % noise in output
xmax = ones(settings.dim_x,1); xmin=-xmax;                                % range of data

settings.projection = 'state_independant';                        % {'state_independant' 'state_dependant'}
settings.task_policy_type = 'random';                           % {'random'}
settings.null_policy_type = 'limit_cycle';                 % {'limit_cycle' 'linear_attractor' 'linear'}
settings.control_space    = 'end_effector';                     % control space in end_effector

fprintf('< Dim_x             = %d                                                   >\r',settings.dim_x);
fprintf('< Dim_u             = %d                                                   >\r',settings.dim_u);
fprintf('< Dim_r             = %d                                                   >\r',settings.dim_r);
fprintf('< Dim_k             = %d                                                   >\r',settings.dim_k);
fprintf('< Constraint        = %s                                                   >\r',settings.projection);
fprintf('< Null_policy_type  = %s                                                   >\r',settings.null_policy_type);
fprintf('< Task_policy_type  = %s                                                   >\r',settings.task_policy_type);

%% NULL SPACE POLICY GENERATION
switch settings.null_policy_type
    case 'limit_cycle'
        radius = 0.75;                                          % radius of attractor
        qdot   = 1.0;                                           % angular velocity
        w      = 1.0;                                           % time scaling factor
        f_n = @(x)(-w*[(radius-x(1,:).^2-x(2,:).^2).*x(1,:) - x(2,:)*qdot;(radius-x(1,:).^2-x(2,:).^2).*x(2,:) + x(1,:)*qdot]);
    case 'linear_attractor'
        target = [0 0]';
        f_n    = @(x) settings.null.alpha .* (target - x);      % nullspace policy
    case 'linear'
        w    = [1 2;3 4;-1 0];
        f_n    = @(x)(([x;ones(1,size(x,2))]'*w)');
end
settings.f_n = f_n;

%% TASK SPACE POLICY GENERATION
f_b = @(N)(2*rand(1,1,N)-1);
% f_b    = @(x) settings.null.alpha .* (randi([-2,2]) - x);      % nullspace policy
settings.f_b = f_b;

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
model = [];
Ntr  = 500;
Nte   = 500;
A_ = orth( rand(2, 1) )';
f_A = @(q)(A_);                       % random constraint
model.num_basis = 16 ;                % define the number of radial basis functions
settings.f_A = f_A;
settings.grid_on = 1;
fprintf('\n< Generating training dataset for learning null space components  ...     >\r');
settings.N = Ntr;
Dtr = generate_data_ccl(settings);
Xtr = Dtr.X; Ytr = Dtr.Y;
Ftr = Dtr.F; Atr = Dtr.A; Ptr = Dtr.P;
NStr = Dtr.NS; TStr = Dtr.TS;
fprintf(1,'#Data (train): %5d, \r',Ntr);

% generate test data
fprintf('< Generating testing dataset for learning null space components   ...     >\n');
settings.N = Nte;
Dte = generate_data_ccl(settings);
Xte = Dte.X; Yte = Dte.Y;
Fte = Dte.F; Ate = Dte.A; Pte = Dte.P;
TSte = Dte.TS; NSte = Dte.NS;
fprintf(1,'#Data (test): %5d, \n',Nte);

% generate visualisation data
try
    error
catch
    Ngp  = 10; Nv = Ngp^settings.dim_x;
    [X1v X2v] = ndgrid(linspace(xmin(1),xmax(1),Ngp),linspace(xmin(2),xmax(2),Ngp)); Xv = [X1v(:),X2v(:)]';
    Fv = f_n(Xv);
    for i = 1:Nv
        Av  = f_A(Xv(:,i)) ;
        P  = eye(2) - pinv(Av)*Av;
        NSv(:,i)= P*Fv(:,i) ;
    end
end

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
NSpv  = f_ncl (Xv)  ;

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

% visualisation
figNo = 2;
figure(figNo),clf,hold on,grid on,box on
s = .1;
h(1) = quiver(Xtr(1,:), Xtr(2,:), s*Ytr (1,:), s*Ytr (2,:), 0, 'Color', ctr,'LineWidth',lwtr); % training data
h(2) = quiver(Xv (1,:) , Xv(2,:), s*NSpv(1,:), s*NSpv(2,:), 0, 'Color', cpv,'LineWidth',lwpv); % learnt nullspace component
h(3) = quiver(Xv (1,:) , Xv(2,:), s*NSv (1,:), s*NSv (2,:), 0, 'Color',  cv,'LineWidth', lwv); % true nullspace component
legend(h,'Data','True','Estimated','Location','Best');legend boxoff
title('Learning null space component');
axis tight
clear figNo
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
pause();
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                            LEARN CONSTRAINTS                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n\n<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('< SECTION 3:       LEARNING NULL SPACE CONSTRAINTS                        >\r');
fprintf('< In this section,we will start addressing state independant constraint A >\r');
fprintf('< and then state dependant constraint A(x). The exploration policy will   >\r');
fprintf('< change the performance of the learnt constraint                         >\r');
fprintf('< The cost function to be minimised is:                                   >\r');
fprintf('<                           E[A] = min(sum||A*Un||^2)                     >\n');
fprintf('\n< For details please refer to:                                            >\r');
fprintf('< H.-C. Lin, M. Howard, and S. Vijayakumar. IEEE International Conference >\r');
fprintf('< Robotics and Automation, 2015                                           >\n');
settings.grid_on = 0;
settings.task_policy_type = ' ';
settings.learn_nc       = 0;
if strcmp(settings.projection,'state_independant')
    fprintf(1,'\n< Start learning state independant null space projection   ...            > \n');
    % generate visualisation data
    settings.dim_n          = 20 ;                          % number of steps in each trajactory
    settings.nTraj          = 50 ;                          % number of trajectories
    settings.dim_exp        = 1 ;                           % number of experiment to repeat
    settings.learn_alpha    = 1;
    A_                      = orth( rand(2, 1) )';
    P                       = eye(2) - pinv(A_)*A_;
    f_alpha                 = @(q)(A_);                     % random constraint
    settings.f_alpha        = f_alpha;
    Dtr                     = [];
    Dte                     = [];
    for i = 1:settings.dim_exp
        % generating training dataset
        fprintf('\n< Generating training dataset for learning constraints    ...             >\r');
        Dtr = generate_data_ccl(settings);
        Xtr = Dtr.X ; Ytr = Dtr.U ; TStr = Dtr.TS ; NStr = Dtr.NS ; Ftr = Dtr.F ;
        fprintf(1,'#Data (train): %5d, \r',Ntr);
        % generating testing data
        fprintf('< Generating testing dataset for learning constraints     ...             >\r');
        Dte = generate_data_ccl (settings) ;
        Xte = Dte.X ; Yte = Dte.U ; TSte = Dtr.TS ; NSte = Dte.NS ; Fte = Dte.F ;
        fprintf(1,'#Data (test): %5d, \n',Nte);
        fprintf(1,'\n< Experiment %d \n', i) ;
        fprintf(1,'\t Dimensionality of action space: %d \n', settings.dim_u) ;
        fprintf(1,'\t Dimensionality of task space:   %d \n', settings.dim_k) ;
        fprintf(1,'\t Dimensionality of null space:   %d \n', settings.dim_u-settings.dim_k) ;
        fprintf(1,'\t Size of the training data:      %d \n', size(Xtr,2)) ;
        
        model  = learn_nhat (NStr);
        
        fprintf(1,'True projection:\n w =\n'),      disp(P);
        fprintf(1,'Estimated projection:\n wp =\n'),disp(model.P)
        
        % make prediction
        fp   = @(f)predict_proj (f,model);
        Yptr = fp(Ftr);
        Ypte = fp(Fte);
        Ypv  = fp(Fv) ;
        
        fprintf(1,'\n Result %d \n') ;
        fprintf(1,'\t ===============================\n' ) ;
        fprintf(1,'\t       |    NPPE   |   NPOE \n' ) ;
        fprintf(1,'\t -------------------------------\n' ) ;
        nPPE = get_ppe(Ytr, model.P, Ftr) ;
        nPOE = get_poe(Ytr, model.P, Ftr) ;
        fprintf(1,'\t Train |  %4.2e | %4.2e  \n', nPPE,  nPOE ) ;
        nPPE = get_ppe(Yte, model.P, Fte) ;
        nPOE = get_poe(Yte, model.P, Fte) ;
        fprintf(1,'\t Test  |  %4.2e | %4.2e  \n', nPPE,  nPOE ) ;
        fprintf(1,'\t ===============================\n' ) ;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(settings.projection, 'state_dependant')
    
    fprintf(1,'\n< Start learning state dependant null space projection   ...              > \n');
    settings.dim_b          = 10 ;                          % dimensionality of the kernel function. 16 is normally enough for a 2D problem
    settings.dim_n          = 20 ;                          % number of steps in each trajactory
    settings.nTraj          = 50 ;                          % number of trajectories
    settings.dim_exp        = 1 ;                           % number of experiment to repeat
    settings.learn_alpha    = 1;
    Ntr                     = settings.dim_n*settings.nTraj;
    Nte                     = Ntr;
    Dtr                     = [];
    Dte                     = [];
    for i = 1:settings.dim_exp
        fprintf('< Generating training dataset for learning constraints     ...             >\r');
        Dtr = generate_data_ccl(settings);
        Xtr = Dtr.X ; Ytr = Dtr.U ; TStr = Dtr.TS ; NStr = Dtr.NS ; Ftr = Dtr.F ;
        fprintf(1,'#Data (train): %5d, \r',Ntr);
        % generating testing data
        fprintf('< Generating testing dataset learning constraints      ...                 >\r');
        Dte = generate_data_ccl (settings) ;
        Xte = Dte.X ; Yte = Dte.U ; TSte = Dte.TS ; NSte = Dte.NS ; Fte = Dte.F ;
        fprintf(1,'#Data (test): %5d, \n',Nte);
        fprintf(1,'\n Experiment %d \n', i) ;
        fprintf(1,'\t Dimensionality of action space: %d \n', settings.dim_u) ;
        fprintf(1,'\t Dimensionality of task space:   %d \n', settings.dim_k) ;
        fprintf(1,'\t Dimensionality of null space:   %d \n', settings.dim_u-settings.dim_k) ;
        fprintf(1,'\t Dimensionality of kernel:       %d \n', settings.dim_b) ;
        fprintf(1,'\t Size of the training data:      %d \n', size(Xtr,2)) ;
        %         fprintf(1,'\t Random constraint:  [%4.2f, %4.2f] \n', settings.alpha(1), settings.alpha(2)) ;
        fprintf(1,'\n Learning state-dependent constraint vectors... \n') ;
        
        model_alpha_ccl  = learn_alpha_ccl (NStr, Xtr, settings);
        
        ppe.alpha_ccl.tr = get_ppe_alpha (model_alpha_ccl.f_proj, Xtr, Ftr, NStr) ;
        ppe.alpha_ccl.te = get_ppe_alpha (model_alpha_ccl.f_proj, Xte, Fte, NSte) ;
        poe.alpha_ccl.tr = get_poe_alpha (model_alpha_ccl.f_proj, Xtr, NStr) ;
        poe.alpha_ccl.te = get_poe_alpha (model_alpha_ccl.f_proj, Xte, NSte) ;
        
        fprintf(1,'\n Result %d \n') ;
        fprintf(1,'\t ===============================\n' ) ;
        fprintf(1,'\t       |    NPPE   |   NPOE \n' ) ;
        fprintf(1,'\t -------------------------------\n' ) ;
        fprintf(1,'\t Train |  %4.2e | %4.2e  \n', ppe.alpha_ccl.tr,  poe.alpha_ccl.tr ) ;
        fprintf(1,'\t Test  |  %4.2e | %4.2e  \n', ppe.alpha_ccl.te,  poe.alpha_ccl.te ) ;
        fprintf(1,'\t ===============================\n' ) ;
    end
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
fprintf('<                  E[Pi] = min(sim||U_ns - U_ns^2*Pi||^2)                 >\r');
fprintf('< To achiev a better unconstrained null space policy, more constraints    >\r');
fprintf('< need to be demonstrated in the training dataset                         >\n');
fprintf('\n< For details please refer to:                                            >\r');
fprintf('< Howard, Matthew, et al. "A novel method for learning policies from      >\r');
fprintf('< variable constraint data." Autonomous Robots 27.2 (2009): 105-121.      >\n');
fprintf('\n< Start learning null space policy   ...                                 > \n');
Dtr     = [];
Dte     = [];
settings.grid_on = 1;
if strcmp(settings.projection, 'state_independant')
    f_A = @(q)(orth( rand(2, 1) )');                       % random constraint
elseif strcmp(settings.projection, 'state_dependant')
    f_A = @(q)([2*randi([-10,10])*q(1),-1]);            % parabola constraint
end
settings.f_A = f_A;
Ntr          = 500;
Nte          = 500;
fprintf('\n< Generating training dataset for learning null space constraints  ...    >\n');
settings.N = Ntr;
Dtr = generate_data_ccl(settings);
Xtr = Dtr.X; Ytr = Dtr.Y;
Ftr = Dtr.F; Atr = Dtr.A; Ptr = Dtr.P;
NStr = Dtr.NS; TStr = Dtr.TS;
fprintf(1,'#Data (train): %5d, \r',Ntr);

% generate test data
fprintf('\n< Generating testing dataset for learning null space constraints   ...    >\n');
settings.N = Nte;
Dte = generate_data_ccl(settings);
Xte = Dte.X; Yte = Dte.Y;
Fte = Dte.F; Ate = Dte.A; Pte = Dte.P;
TSte = Dte.TS; NSte = Dte.NS;

fprintf(1,'#Data (test): %5d, \n',Nte);

try
    error
catch
    fprintf(1,'\n< Start learning Null space policy with parametric model  ...             > \n');
    Ngp  = 10; Nv = Ngp^settings.dim_x;
    [X1v X2v] = ndgrid(linspace(xmin(1),xmax(1),Ngp),linspace(xmin(2),xmax(2),Ngp)); Xv = [X1v(:),X2v(:)]';
    Fv = f_n(Xv);
end
% set up regression model
model = [];
model.w = [];
cmax = xmax+.1;
cmin = xmin-.1;
Ngp  = 10; Nc = Ngp^settings.dim_x;
[c1,c2] = ndgrid(linspace(cmin(1),cmax(1),Ngp),linspace(cmin(2),cmax(2),Ngp)); c = [c1(:),c2(:)]';
s2   = 0.5;
model.phi = @(x)phi_gaussian_rbf ( x, c, s2 );
%     model.phi = @(x)phi_linear ( x );

% train parametric model
model = learn_ccl_pi(Xtr,NStr,model); fp = @(x)predict_linear(x,model);

% predict training data
Fptr = fp(Xtr);

% compute training error
NUPEtr = get_nupe(Ftr,Fptr);
fprintf(1,'NUPE (train) = %5.2e, ',NUPEtr);
NCPEtr = get_ncpe(NStr,Fptr,Ptr);
fprintf(1,'NCPE (train) = %5.2e, ',NCPEtr);

% predict test data
Fpte = fp(Xte);

% compute test error
NUPEte = get_nupe(Fte,Fpte);
fprintf(1,'NUPE (test) = %5.2e, ',NUPEte);
NCPEte = get_ncpe(NSte,Fpte,Pte);
fprintf(1,'NCPE (test) = %5.2e',NCPEte);
fprintf(1,'\n');

% visualisation
figNo = 4;
figure(figNo),clf,hold on,grid on,box on
% visualise the training data
s = .1;
h(1)=quiver(Xtr(1,:),Xtr(2,:),s* NStr(1,:),s* NStr(2,:),0,'Color', ctr,'LineWidth', lwtr);
% visualise the fit
% predict visualisation data
Fpv = fp(Xv);
% visualise the fit
h(2)=quiver(Xv(1,:),Xv(2,:),s*Fpv(1,:),s*Fpv(2,:),0,'Color',cpv,'LineWidth',lwpv);
h(3)=quiver(Xv(1,:),Xv(2,:),s* Fv(1,:),s* Fv(2,:),0,'Color', cv,'LineWidth', lwv);
legend(h,'Data','Estimated','True','Location','Best');legend boxoff
title('Learning null space policy with linear parametric model')
axis tight
pause();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                              LEARN LOCALLY WEIGHTED LINEAR POLICY MODEL                            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(1,'\n< Start learning Null space policy with locally weighted  model   ...     > \n');
model = [];
Ngp  = 10; Nc = Ngp^settings.dim_x;
[c1,c2] = ndgrid(linspace(cmin(1),cmax(1),Ngp),linspace(cmin(2),cmax(2),Ngp)); c = [c1(:),c2(:)]';
s2 = 0.005;
model = [];
model.W   = @(x)phi_gaussian_rbf( x, c, s2 );
 model.phi = @(x)phi_linear( x );
% model.phi = @(x)phi_gaussian_rbf ( x, c, s2 );

% train the model
model = learn_lwccl(Xtr,NStr,model); fp = @(x)predict_local_linear(x,model);

% predict training data
Fptr = fp(Xtr);

% compute training error
NUPEtr = get_nupe(Ftr,Fptr);
fprintf(1,'NUPE (train) = %5.2e, ',NUPEtr);
NCPEtr = get_ncpe(NStr,Fptr,Ptr);
fprintf(1,'NCPE (train) = %5.2e, ',NCPEtr);

% predict test data
Fpte = fp(Xte);

% compute test error
NUPEte = get_nupe(Fte,Fpte);
fprintf(1,'NUPE (test) = %5.2e, ',NUPEte);
NCPEte = get_ncpe(NSte,Fpte,Pte);
fprintf(1,'NCPE (test) = %5.2e',NCPEte);
fprintf(1,'\n');

% visualisation
figNo = 5;
figure(figNo),clf,hold on,grid on,box on
% visualise the training data
s = .1;
h(1)=quiver(Xtr(1,:),Xtr(2,:),s* NStr(1,:),s* NStr(2,:),0,'Color', ctr,'LineWidth', lwtr);
% visualise the fit
% predict visualisation data
Fpv = fp(Xv);
% visualise the fit
h(2)=quiver(Xv(1,:),Xv(2,:),s*Fpv(1,:),s*Fpv(2,:),0,'Color',cpv,'LineWidth',lwpv);
h(3)=quiver(Xv(1,:),Xv(2,:),s* Fv(1,:),s* Fv(2,:),0,'Color', cv,'LineWidth', lwv);
legend(h,'Data','Estimated','True','Location','Best');legend boxoff
title('Learning null space policy with locally weighted parametric model')
axis tight
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
pause();
close all;
end