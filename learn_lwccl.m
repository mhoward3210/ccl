
function model = learn_ccl(X,Y,model)

[dimY N] = size(Y);

% find normalised Y
r = sum(Y.^2,1).^0.5;
YN = Y./repmat(r,dimY,1);

% find feature vectors
Phi      = model.phi(X);
dimPhi   = size(Phi,1); % get feature dimensionality

% find weights
W        = model.W(X);
Nc       = size(W,1);   % get no. centres

% train each local model
for nc=1:Nc
	WPhi=repmat(W(nc,:),dimPhi,1).*Phi;

	% construct Jacobian
	YPhit = Y*WPhi';
	g = YPhit(:);

	% construct Hessian
	H = zeros(dimY*dimPhi);
	for n=1:N
	YNPhit = YN(:,n)*Phi(:,n)';
	v(:,n) = YNPhit(:);
	H = H + W(nc,n)*v(:,n)*v(:,n)';
	end

	% do eigendecomposition for inversion
	%[V,D] = eig(H+1e-6*eye(size(H)));
	[V,D] = eig(H);
	ev = diag(D);
	ind = find(ev>1e-6);
	V1=V(:,ind);
	pinvH1 = V1*diag(ev(ind).^-1)*V1';
	model.w(:,:,nc)=reshape(pinvH1*g,dimY,dimPhi)';

end

