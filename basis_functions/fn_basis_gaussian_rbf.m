%{ 
Gaussian radial basis function

input
    X: input data
    C: centre
    s2: variance
   
out: 
    BX: gaussian radial basis function of X
%}
function BX = fn_basis_gaussian_rbf ( X, C, s2 )    
    numC= size(C,2) ;
    numX= size(X,2) ;
    C2  = sum(C.^2) ;
    X2  = sum(X.^2) ;    
    D   = repmat( C2',1, numX ) + repmat( X2, numC, 1) - 2*C'*X;      
    BX  = exp(-0.5/s2*D) ;
end
