%{
calculate the euclidean distance between two datasets A and B
input
    A: a dataset with N data points
    B: a dataset with M data points
output
    D: euclidean distance between A and B. D is a MxN matrix
%}
function D = distances (A, B) 
    numA= size(A,2) ;
    numB= size(B,2) ;
    A2  = sum(A.^2) ;
    B2  = sum(B.^2) ;    
    D   = repmat( A2',1, numB ) + repmat( B2, numA, 1) - 2*A'*B ;   
end
