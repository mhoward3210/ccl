% Script to demonstrate multilinear regression on a 2-D linear data set.

clear all
rand('seed',1),randn('seed',1)
figNo=2;
ctr = .9*ones(1,3); lwtr = 1; % colour, linewidth for training data
cv  = [0  0  0];    lwv  = 1; % colour, linewidth for visualisation data
cpv = [1 .5 .5];    lwpv = 2; % colour, linewidth for visualisation data predictions

% generative model
dimX = 2;
dimY = 2;
w = [1 2;3 4;-1 0];
f = @(x)(([x;ones(1,size(x,2))]'*w)');
s2y  = .01; % noise in output

xmax = ones(dimX,1); xmin=-xmax; % range of data

% generate data
% generate training data
try   % load data, if possible
Dtr=load_data_ccl(['D_tr_',mfilename]);
Xtr=Dtr.X; Ytr=Dtr.Y; Ntr=Dtr.N;
Ftr=Dtr.F; Atr=Dtr.A; Ptr=Dtr.P;
catch
Ntr  = 500;
Xtr  = repmat(xmax-xmin,1,Ntr).*rand(dimX,Ntr)+repmat(xmin,1,Ntr);
Ftr  = f(Xtr);
Atr  = rand(1,2,Ntr);
for n=1:Ntr
	Ptr(:,:,n) = eye(2) - pinv(Atr(:,:,n))*Atr(:,:,n);
	Ytr(:,n) = Ptr(:,:,n)*Ftr(:,n)+s2y*randn(dimY,1);
end
Dtr.X = Xtr; Dtr.Y = Ytr; Dtr.N = Ntr;
Dtr.F = Ftr; Dtr.A = Atr; Dtr.P = Ptr;
save_data_ccl(['data/D_tr_',mfilename],Dtr);
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
Ate  = rand(1,2,Nte);
for n=1:Nte
	Pte(:,:,n) = eye(2) - pinv(Ate(:,:,n))*Ate(:,:,n);
	Yte(:,n) = Pte(:,:,n)*Fte(:,n)+s2y*randn(dimY,1);
end
Dte.X = Xte; Dte.Y = Yte; Dte.N = Nte;
Dte.F = Fte; Dte.A = Ate; Dte.P = Pte;
end
fprintf(1,'#Data (test): %5d, ',Nte);   
% generate visualisation data
try
error
catch
Ngp  = 10; Nv = Ngp^dimX;
[X1v X2v] = ndgrid(linspace(xmin(1),xmax(1),Ngp),linspace(xmin(2),xmax(2),Ngp)); Xv = [X1v(:),X2v(:)]';
Fv = f(Xv);
end

% set up regression model
model.w = [];
cmax = xmax+.1;
cmin = xmin-.1;
Ngp  = 10; Nc = Ngp^dimX;
[c1,c2] = ndgrid(linspace(cmin(1),cmax(1),Ngp),linspace(cmin(2),cmax(2),Ngp)); c = [c1(:),c2(:)]';
s2   = 1.0;
model.phi = @(x)fn_basis_gaussian_rbf ( x, c, s2 );

% train the model
model = learn_ccl(Xtr,Ytr,model); fp = @(x)predict_linear(x,model);

% predict training data
Fptr = fp(Xtr);

% compute training error
NUPEtr = get_nupe(Ftr,Fptr);
fprintf(1,'NUPE (train) = %5.3f, ',NUPEtr);   
NCPEtr = get_ncpe(Ytr,Fptr,Ptr);
fprintf(1,'NCPE (train) = %5.3f, ',NCPEtr);   

% predict test data
Fpte = fp(Xte);

% compute test error
NUPEte = get_nupe(Fte,Fpte);
fprintf(1,'NUPE (test) = %5.3f, ',NUPEte);   
NCPEte = get_ncpe(Yte,Fpte,Pte);
fprintf(1,'NCPE (test) = %5.3f',NCPEte);   
fprintf(1,'\n');   

% visualisation
figure(figNo),clf,hold on,grid on,box on
% visualise the training data
s = .1; 
h(1)=quiver(Xtr(1,:),Xtr(2,:),s* Ytr(1,:),s* Ytr(2,:),0,'Color', ctr,'LineWidth', lwtr);
% visualise the fit
% predict visualisation data
Fpv = fp(Xv);
% visualise the fit                                                 
h(2)=quiver(Xv(1,:),Xv(2,:),s*Fpv(1,:),s*Fpv(2,:),0,'Color',cpv,'LineWidth',lwpv);
h(3)=quiver(Xv(1,:),Xv(2,:),s* Fv(1,:),s* Fv(2,:),0,'Color', cv,'LineWidth', lwv);
legend(h,'Data','True','Estimated','Location','Best');legend boxoff
axis tight

clear figNo
