%{ 
Gaussian radial basis function

input
    X: input data
    c: centre
    s2: variance
   
out: 
    BX: gaussian radial basis function of X
%}
function BX = fn_basis_gaussian_rbf ( x, c, s2 )
    D  = distances(c,x);
    BX = exp(-(0.5/s2)*D);
end

function D = distances(A,B)
    m = size(A,2);
    n = size(B,2);
    D = repmat( (A.^2)',1,n) + repmat( (B.^2), m,1) - 2*A'*B;
end
