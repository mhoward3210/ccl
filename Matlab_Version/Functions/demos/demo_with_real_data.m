function demo_with_real_data
% demo_with_real_data
% A demonstration of using CCL library with real data from Trakstar sensor.
% The data is x,y,z position in the operational space. The operator
% attached one sensor on the finger tip and slided on different surface
% constraints.
clear all;clc;close all; rng('default');
fprintf('<=========================================================================>\r');
fprintf('<=========================================================================>\r');
fprintf('<===========   Constraint Consistent Learning Library    =================>\r');
fprintf('< This demo will demonstrate the CCL toolbox using real data from Trakstar>\r');
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
fprintf('< SECTION 2:       LEARNING NULL SPACE CONSTRAINTS                        >\r');
fprintf('< Configuration options:                                                  >\r');
fprintf('< Constraints:                                                            >\r');
fprintf('<            State independant:                                           >\r');
fprintf('<                              linear(random)                             >\r');
fprintf('<              State dependant:                                           >\r');
fprintf('<                              parabola                                   >\r');
fprintf('< Null space policy:                                                      >\r');
fprintf('<                  circular policy demonstrated by human                  >\r');
fprintf('<=========================================================================>\r');
fprintf('<=========================================================================>\r');
fprintf('<=========================================================================>\n\n\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('< SECTION 1:       PARAMETER CONFIGURATION                                >\r');
fprintf('\n< User specified configurations are:                                      >\r');
%% GENERATIVE MODEL PARAMETERS
ctr = 1*ones(1,3);                                              % colour for training data
cte  = [0  0  0];                                               % colour for testing data
settings.dim_x          = 3 ;                                   % dimensionality of the state space
settings.dim_u          = 3 ;                                   % dimensionality of the action space
settings.dim_r          = 3 ;                                   % dimensionality of the task space
settings.dim_k          = 1 ;                                   % dimensionality of the constraint
settings.dt             = 0.02;                                 % time step
settings.projection = 'state_dependant';                      % {'state_independant' 'state_dependant'}
settings.null_policy_type = 'circle';                           % {'circle'}

fprintf('< Dim_x             = %d                                                   >\r',settings.dim_x);
fprintf('< Dim_u             = %d                                                   >\r',settings.dim_u);
fprintf('< Dim_r             = %d                                                   >\r',settings.dim_r);
fprintf('< Dim_k             = %d                                                   >\r',settings.dim_k);
fprintf('< Constraint        = %s                                                   >\r',settings.projection);
fprintf('< Null_policy_type  = %s                                                   >\r',settings.null_policy_type);

fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
pause();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                            LEARN CONSTRAINTS                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n\n<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('< SECTION 2:       LEARNING NULL SPACE CONSTRAINTS                        >\r');
fprintf('< In this section,we will start addressing state independant constraint A >\r');
fprintf('< and then state dependant constraint A(x). The exploration policy will   >\r');
fprintf('< change the performance of the learnt constraint                         >\r');
fprintf('< The cost function to be minimised is:                                   >\r');
fprintf('<                           E[A] = min(sum||A*Un||^2)                     >\n');
fprintf('\n< For details please refer to:                                            >\r');
fprintf('< H.-C. Lin, M. Howard, and S. Vijayakumar. IEEE International Conference >\r');
fprintf('< Robotics and Automation, 2015                                           >\n');

if strcmp(settings.projection,'state_independant')
    fprintf(1,'\n< Start learning state independant null space projection   ...            > \n');
    N_tr                    = 0.7;
    N_te                    = 0.3;
    options.dim_b           = 5;
    options.dim_r           = 3;
    % generating training and testing dataset
    fprintf('< Generating training and testingdataset for learning constraints    ...  >\r');
    Data = ccl_data_load('planer_circle');
    [ind_tr,ind_te] = dividerand(Data.N,N_tr,N_te,0) ;
    Xtr = Data.X(:,ind_tr);Ytr = Data.Y(:,ind_tr);
    Xte = Data.X(:,ind_te);Yte = Data.Y(:,ind_te);
    fprintf(1,'#Data (train): %5d, \r',size(Xtr,2));
    fprintf(1,'#Data (test): %5d, \n',size(Xte,2));
    fprintf(1,'\t Dimensionality of action space: %d \n', settings.dim_u) ;
    fprintf(1,'\t Dimensionality of task space:   %d \n', settings.dim_k) ;
    fprintf(1,'\t Dimensionality of null space:   %d \n', settings.dim_u-settings.dim_k) ;
    fprintf(1,'\t Size of the training data:      %d \n', size(Xtr,2)) ;
    fprintf(1,'\n Learning state-independent constraint vectors  ... \n') ;
    
    model  = ccl_learna_alpha (Ytr,Xtr,options);
    fprintf(1,'\n Result %d \n') ;
    fprintf(1,'\t ===============================\n' ) ;
    fprintf(1,'\t       |    NPOE    VPOE    UPOE\n' ) ;
    fprintf(1,'\t -------------------------------\n' ) ;
    [nPOE,vPOE,uPOE] = ccl_error_poe_alpha (model.f_proj, Xtr, Ytr) ;
    fprintf(1,'\t Train |  %4.2e  %4.2e  %4.2e \n',  nPOE, sum(vPOE), uPOE ) ;
    nPOE = ccl_error_poe_alpha (model.f_proj, Xte, Yte) ;
    fprintf(1,'\t Test  |  %4.2e  %4.2e  %4.2e  \n',  nPOE, sum(vPOE), uPOE ) ;
    fprintf(1,'\t ===============================\n' ) ;
    figure;scatter3(Xtr(1,:),Xtr(2,:),Xtr(3,:),'filled','MarkerEdgeColor','k',...
        'MarkerFaceColor',ctr);hold on;
    scatter3(Xte(1,:),Xte(2,:),Xte(3,:),'MarkerEdgeColor','k',...
        'MarkerFaceColor',cte);
    zlim([0,1]);xlabel('x');ylabel('y');zlabel('z');title('Training (black) & Testing (white) data visualisation');
    for n = 1:size(Yte,2)
        NS_p(:,n) = model.f_proj(Xte(:,n)) * Yte(:,n) ;
    end
    figure;plot(real(NS_p(1,:)));hold on;plot(Yte(1,:)); legend('prediction','true observation');title('Prediciton VS True observation'); xlabel('Time step'); ylabel('Y1');
    figure;plot(real(NS_p(2,:)));hold on;plot(Yte(2,:)); legend('prediction','true observation');title('Prediciton VS True observation'); xlabel('Time step'); ylabel('Y2');
    figure;plot(real(NS_p(3,:)));hold on;plot(Yte(3,:)); legend('prediction','true observation');title('Prediciton VS True observation'); xlabel('Time step'); ylabel('Y3');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(settings.projection, 'state_dependant')
    fprintf(1,'\n< Start learning state dependant null space projection   ...              > \n');
    N_tr                    = 0.7;
    N_te                    = 0.3;
    options.dim_b           = 5;
    options.dim_r           = 3;
    % generating training dataset
    fprintf('< Generating training and testingdataset for learning constraints    ... >\r');
    Data = ccl_data_load('curve_circle');
    [ind_tr,ind_te] = dividerand(Data.N,N_tr,N_te,0) ;
    Xtr = Data.X(:,ind_tr);Ytr = Data.Y(:,ind_tr);
    Xte = Data.X(:,ind_te);Yte = Data.Y(:,ind_te);
    fprintf(1,'#Data (train): %5d, \r',size(Xtr,2));
    fprintf(1,'#Data (test): %5d, \n',size(Xte,2));
    fprintf(1,'\t Dimensionality of action space: %d \n', settings.dim_u) ;
    fprintf(1,'\t Dimensionality of task space:   %d \n', settings.dim_k) ;
    fprintf(1,'\t Dimensionality of null space:   %d \n', settings.dim_u-settings.dim_k) ;
    fprintf(1,'\t Size of the training data:      %d \n', size(Xtr,2)) ;
    fprintf(1,'\n Learning state-dependent constraint vectors... \n') ;
    
    model  = ccl_learna_alpha (Ytr,Xtr,options);
    
    fprintf(1,'\n Result %d \n') ;
    fprintf(1,'\t ===============================\n' ) ;
    fprintf(1,'\t       |    NPOE    VPOE    UPOE\n' ) ;
    fprintf(1,'\t -------------------------------\n' ) ;
    [nPOE,vPOE,uPOE] = ccl_error_poe_alpha (model.f_proj, Xtr, Ytr) ;
    fprintf(1,'\t Train |  %4.2e  %4.2e  %4.2e \n',  nPOE, sum(vPOE), uPOE ) ;
    nPOE = ccl_error_poe_alpha (model.f_proj, Xte, Yte) ;
    fprintf(1,'\t Test  |  %4.2e  %4.2e  %4.2e  \n',  nPOE, sum(vPOE), uPOE ) ;
    fprintf(1,'\t ===============================\n' ) ;
    figure;scatter3(Xtr(1,:),Xtr(2,:),Xtr(3,:),'filled','MarkerEdgeColor','k',...
        'MarkerFaceColor',ctr);hold on;
    scatter3(Xte(1,:),Xte(2,:),Xte(3,:),'MarkerEdgeColor','k',...
        'MarkerFaceColor',cte);
    xlabel('x');ylabel('y');zlabel('z');title('Training (black) & Testing (white) data visualisation');
    for n = 1:size(Yte,2)
        NS_p(:,n) = model.f_proj(Xte(:,n)) * Yte(:,n) ;
    end
    figure;plot(real(NS_p(1,:)));hold on;plot(Yte(1,:)); legend(['prediction','true observation']);title('Prediciton VS True observation'); xlabel('Time step'); ylabel('Y1');
    figure;plot(real(NS_p(2,:)));hold on;plot(Yte(2,:)); legend(['prediction','true observation']);title('Prediciton VS True observation'); xlabel('Time step'); ylabel('Y2');
    figure;plot(real(NS_p(3,:)));hold on;plot(Yte(3,:)); legend(['prediction','true observation']);title('Prediciton VS True observation'); xlabel('Time step'); ylabel('Y3');
end
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
fprintf('<=========================================================================>\n');
pause();
close all;
end