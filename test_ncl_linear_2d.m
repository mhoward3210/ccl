% Script to demonstrate nullspace component learning on a 2-D linear data
% set with a consistent constraint

clear all
rand('seed',1),randn('seed',1)
figNo=1;
ctr = .9*ones(1,3); lwtr = 1; % colour, linewidth for training data
cv  = [0  0  0];    lwv  = 1; % colour, linewidth for visualisation data
cpv = [1 .5 .5];    lwpv = 2; % colour, linewidth for visualisation data predictions

% generative model
dimX = 2 ;                                  % dimensionality of the state space
dimY = 2 ;                                  % dimensionality of the action space
w    = [1 2;3 4;-1 0];                      
f    = @(x)(([x;ones(1,size(x,2))]'*w)');   % nullspace policy
s2y  = .01;                                 % noise in output

xmax=ones(dimX,1); xmin=-xmax;              % range of data
A   = rand(1,2) ;                           % constraint matrix
P   = eye(2) - pinv(A)*A ;                  % projection matrix

% =============  generate data ============================================

% generate training data
try   % load data, if possible
    Dtr=load_data_ccl(['data/D_tr_',mfilename]);
    Xtr=Dtr.X; Ytr=Dtr.Y; Ntr=Dtr.N;
    Ftr=Dtr.F; Atr=Dtr.A; Ptr=Dtr.P;
    NStr=Dtr.NS ; TStr= Dtr.TS ;
catch
    Ntr  = 500;
    Xtr  = repmat(xmax-xmin,1,Ntr).*rand(dimX,Ntr)+repmat(xmin,1,Ntr);
    Ftr  = f(Xtr);
    %Atr  = rand(1,2,Ntr);
    Btr  = 2*(2*rand(1,1,Ntr)-1);
    for n=1:Ntr        
        Atr(:,:,n)  = A ;
        Ptr(:,:,n)  = P ;
        NStr(:,n)   = Ptr(:,:,n)*Ftr(:,n) ;
        TStr(:,n)   = pinv(Atr(:,:,n))*Btr(:,n) ;
        Ytr(:,n)    = TStr(:,n) + NStr(:,n) + s2y*randn(dimY,1);
    end
    Dtr.X = Xtr; Dtr.Y = Ytr; Dtr.N = Ntr;
    Dtr.F = Ftr; Dtr.A = Atr; Dtr.P = Ptr; 
    Dtr.NS = NStr ; Dtr.TS = TStr ;  
    save_data_ccl(['data/D_tr_',mfilename],Dtr);
end
fprintf(1,'#Data (train): %5d, ',Ntr);   

% generate test data
try
    Dte=load_data_ccl(['data/D_te_',mfilename]);
    Xte=Dte.X; Yte=Dte.Y; Nte=Dte.N;
    Fte=Dte.F; Ate=Dte.A; Pte=Dte.P;
    NSte=Dte.NS ; TSte= Dte.TS ;
catch
    Nte  = 500;
    Xte  = repmat(xmax-xmin,1,Nte).*rand(dimX,Nte)+repmat(xmin,1,Nte);
    Fte  = f(Xte);
    Bte  = 2*(2*rand(1,1,Ntr)-1);  
    for n=1:Nte
        Ate(:,:,n)  = A ;
        Pte(:,:,n)  = P ;
        NSte(:,n)   = Pte(:,:,n)*Fte(:,n) ;
        TSte(:,n)   = pinv(Ate(:,:,n))*Bte(:,n) ;
        Yte(:,n)    = TSte(:,n) + NSte(:,n) + s2y*randn(dimY,1);
    end
    Dte.X = Xte ; Dte.Y = Yte; Dte.N = Nte;
    Dte.F = Fte ; Dte.A = Ate; Dte.P = Pte;    
    Dte.TS = TSte; Dte.NS = NSte ; 
    save_data_ccl(['data/D_te_',mfilename],Dte);
end
fprintf(1,'#Data (test): %5d, ',Nte);   
    
% generate visualisation data
try
    error
catch
    Ngp  = 10; Nv = Ngp^dimX;
    [X1v X2v] = ndgrid(linspace(xmin(1),xmax(1),Ngp),linspace(xmin(2),xmax(2),Ngp)); Xv = [X1v(:),X2v(:)]';
    Fv = f(Xv);
    NSv= P*Fv ;
end

% set up the radial basis functions
model.num_basis = 16 ;                                            % define the number of radial basis functions                      
model.c     = generate_grid_centres (Xtr, model.num_basis) ;      % generate a grid of basis functions
model.s2    = mean(mean(sqrt(distances(model.c, model.c))))^2 ;   % set the variance as the mean distance between centres   
model.phi   = @(x)fn_basis_gaussian_rbf ( x, model.c, model.s2 ); % normalised Gaussian rbfs

% learn the nullspace component
model       = learn_ncl (Xtr, Ytr, model) ;   % learn the model 
f_ncl       = @(x) predict_ncl ( model, x ) ; % set up an inference function

% predict nullspace components
NSptr = f_ncl (Xtr) ;
NSpte = f_ncl (Xte) ;
NSpv  = f_ncl (Xv)  ;

% calculate errors
NUPEtr = get_nupe(NStr, NSptr) ;
NUPEte = get_nupe(NSte, NSpte) ;
YNSptr = get_npe (Ytr,  NSptr) ;
YNSpte = get_npe (Yte,  NSpte) ;
fprintf(1,'NUPE (train) = %5.3f, ', NUPEtr);   
fprintf(1,'NNPE (train) = %5.3f, ', YNSptr);   
fprintf(1,'\n');   
fprintf(1,'NUPE (test)  = %5.3f, ', NUPEte);   
fprintf(1,'NNPE (test)  = %5.3f, ', YNSpte);   
fprintf(1,'\n');   

% visualisation
figure(figNo),clf,hold on,grid on,box on    
    s = .1; 
    h(1) = quiver(Xtr(1,:), Xtr(2,:), s*Ytr (1,:), s*Ytr (2,:), 0, 'Color', ctr,'LineWidth',lwtr); % training data
    h(2) = quiver(Xv (1,:) , Xv(2,:), s*NSpv(1,:), s*NSpv(2,:), 0, 'Color', cpv,'LineWidth',lwpv); % learnt nullspace component
    h(3) = quiver(Xv (1,:) , Xv(2,:), s*NSv (1,:), s*NSv (2,:), 0, 'Color',  cv,'LineWidth', lwv); % true nullspace component
    legend(h,'Data','True','Estimated','Location','Best');legend boxoff
    axis tight
    clear figNo
