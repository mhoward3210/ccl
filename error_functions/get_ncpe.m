
function [ncpe, v, mse] = get_ncpe(F, Fp, P)

% get projected predictions
for n=1:size(P,3) 
	Y (:,n) = P(:,:,n)*F (:,n); 
	Yp(:,n) = P(:,:,n)*Fp(:,n); 
end
[d1 d2 mse] = get_nmse(Y,Yp);
v    = var(F,0,2);         % get variance
ncpe = sum(mse)/sum(v);

