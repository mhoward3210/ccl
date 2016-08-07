%{ 
Calculate normalised Gaussian radial basis function of input X

input
    X: input data
    C: centre
    s2: variance
   
out: 
    BX: normalised gaussian radial basis function of X
%}
function BX = phi_gaussian_rbf ( X, C, s2 )    
    D   = distances(C,X) ;                              % distance between C and X
    BX  = exp(-0.5/s2*D) ;                              % radial basis function of X
    BX  = BX.*repmat(sum(BX).^(-1),size(C,2),1);  % normalise
end
