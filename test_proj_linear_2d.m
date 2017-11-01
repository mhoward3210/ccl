% Script to demonstrate nullspace projection learning on a 2-D linear data
% set
clear all
rand('seed',1),randn('seed',1)
figNo=1;
ctr = .9*ones(1,3); lwtr = 1; % colour, linewidth for training data
cv  = [0  0  0];    lwv  = 1; % colour, linewidth for visualisation data
cpv = [1 .5 .5];    lwpv = 2; % colour, linewidth for visualisation data predictions

% generative model
dimX    = 2;
dimY    = 2;
w       = [1 2;3 4;-1 0];                      
f       = @(x)(([x;ones(1,size(x,2))]'*w)');    % nullspace policy
s2y     = .01;                                  % noise in output
xmax    = ones(dimX,1); xmin=-xmax;             % range of data
   
% ====================== generate data ====================================

% generate training data
try   % load data, if possible
    Dtr=load_data_ccl(['D_tr_',mfilename]);
    Xtr=Dtr.X; Ytr=Dtr.Y; Ntr=Dtr.N;
    Ftr=Dtr.F; Atr=Dtr.A; Ptr=Dtr.P;
      A = Dtr.A(:,:,1) ; P = Dtr.P(:,:,1) ;
catch
    Ntr  = 500;
    Xtr  = repmat(xmax-xmin,1,Ntr).*rand(dimX,Ntr)+repmat(xmin,1,Ntr);
    Ftr  = f(Xtr);  
    A    = orth( rand(2, 1) )' ;
    P    = eye(2) - pinv(A)*A ;  
    for n=1:Ntr
        Atr(:,:,n)  = A ;
        Ptr(:,:,n)  = P ;
        Ytr(:,n)    = Ptr(:,:,n)*Ftr(:,n)+s2y*randn(dimY,1);
    end
    size(Ptr)
    Dtr.X = Xtr; Dtr.Y = Ytr; Dtr.N = Ntr;
    Dtr.F = Ftr; Dtr.A = Atr; Dtr.P = Ptr;
    save_data_ccl(['D_tr_',mfilename],Dtr);
end
fprintf(1,'#Data (train): %5d, ',Ntr);   

% generate test data
try
    Dte=load_data_ccl(['D_te_',mfilename]);
    Xte=Dte.X; Yte=Dte.Y; Nte=Dte.N;
    Fte=Dte.F; Ate=Dte.A; Pte=Dte.P;    
catch
    Nte  = 500;
    Xte  = repmat(xmax-xmin,1,Nte).*rand(dimX,Nte)+repmat(xmin,1,Nte);
    Fte  = f(Xte);   
    for n=1:Nte
        Ate(:,:,n)  = A ;
        Pte(:,:,n)  = P ;
        Yte(:,n)    = Pte(:,:,n)*Fte(:,n)+s2y*randn(dimY,1);
    end
    Dte.X = Xte; Dte.Y = Yte; Dte.N = Nte;
    Dte.F = Fte; Dte.A = Ate; Dte.P = Pte;
    save_data_ccl(['D_te_',mfilename],Dtr);
end
fprintf(1,'#Data (test): %5d, ',Nte);   

% generate visualisation data
try
    error
catch
    Ngp  = 5; Nv = Ngp^dimX;    
    [X1v X2v] = ndgrid(linspace(xmin(1),xmax(1),Ngp),linspace(xmin(2),xmax(2),Ngp)); Xv = [X1v(:),X2v(:)]';
    Fv = f(Xv);   
    Yv = P*Fv ;   
end

% ============= learn the projection matrix ===============================
model = learn_nhat (Ytr);    

fprintf(1,'True projection:\n w =\n'),      disp(P)
fprintf(1,'Estimated projection:\n wp =\n'),disp(model.P)

% make prediction
fp   = @(f)predict_proj (f,model); 
Yptr = fp(Ftr);
Ypte = fp(Fte);
Ypv  = fp(Fv) ;

% compute training error
nPPE = get_ppe(Ytr, model.P, Ftr) ;
nPOE = get_poe(Ytr, model.P, Ftr) ;
fprintf(1,'NPPE (train) = %8.6f,  ', nPPE);   
fprintf(1,'NPOE (train) = %8.6f \n', nPOE);   

% compute test error
nPPE = get_ppe(Yte, model.P, Fte) ;
nPOE = get_poe(Yte, model.P, Fte) ;
fprintf(1,'NPPE (test)  = %8.6f,  ', nPPE);   
fprintf(1,'NPOE (test)  = %8.6f \n', nPOE);   

% visualisation
figure(figNo),clf,hold on,grid on,box on
    % visualise the training data
    s = 1 ; 
    h(1)=quiver(Xtr(1,:),Xtr(2,:),s* Ytr(1,:),s* Ytr(2,:),0,'Color', ctr,'LineWidth', lwtr);                                      
    h(2)=quiver(Xv(1,:),Xv(2,:),s* Yv(1,:),s* Yv(2,:),0,'Color',cpv,'LineWidth',lwpv);
    h(3)=quiver(Xv(1,:),Xv(2,:),s*Ypv(1,:),s*Ypv(2,:),0,'Color', cv,'LineWidth', lwv);
    legend(h,'Training samples','True u','Estimated u','Location','Best');legend boxoff
    xlabel('x') ; ylabel('y')
    axis tight
    axis equal
         
%clear figNo
save(['model_',mfilename])
