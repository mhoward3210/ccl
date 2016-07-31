%{
Compute the normalised projected policy error. This error measures the 
difference between the policy subject to the true constraints, and that of 
the policy subject to the estimated constraints.

input: 
    f_proj: state dependent prediction of the projection matrix
    X_t   : true state
    NS_t  : true null space component 
    Pi    : true null space policy
    
output
    nPPE: normalised projected policy error
    vPPE: variance of the nullspace policy
    uPPE: projected policy error
%}
function [nPPE, vPPE, uPPE] = get_ppe_alpha (N_p, X_t, Pi, NS_t)
    dim_n = size(NS_t,2) ;
    NS_p  = zeros(size(NS_t)) ;
    for n = 1:dim_n       
        NS_p(:,n) = N_p(X_t(:,n)) * Pi(:,n) ;
    end     
    uPPE = mean( sum( (NS_t-NS_p).^2, 1) );
    vPPE = var(NS_t,0,2);   
    nPPE = uPPE / sum(vPPE);    
end
