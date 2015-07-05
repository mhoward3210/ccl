%{
Compute the normalised projected policy error. This error measures the 
difference between the policy subject to the true constraints, and that of 
the policy subject to the estimated constraints.

input: 
    U_t: true observation
    N_p: learnt projection matrix
    P_i: true nullspace policy
output
    nPPE: normalised projected policy error
    vPPE: variance of the nullspace policy
    uPPE: projected policy error
%}
function [nPPE, vPPE, uPPE] = get_ppe(U_t, N_p, Pi)
    dim_n   = size(U_t,2) ;
    U_p     = zeros(size(U_t)) ;
    for n = 1:dim_n       
        U_p(:,n) = N_p*Pi(:,n) ;
    end     
    uPPE = sum((U_t-U_p).^2,2) / dim_n ;
    vPPE = var(Pi,0,2);   
    nPPE = sum(uPPE) / sum(vPPE);
end
