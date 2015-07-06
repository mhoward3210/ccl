%{
Compute the normalised observation error. To evaluate the fit without the 
true nullspace policy, an alternative criterion must be used. This indicates 
the quality of the learnt projection matrix in capturing the image space of 
the observations. i.e., how well the learnt N satisfy Nu = u

input: 
    U_t: true observation
    N_p: learnt projection matrix
    P_i: true nullspace policy
output
    nPPE: normalised projected observation error
    vPPE: variance of the nullspace policy
    uPPE: projected observation error
%}
function [nPOE, vPOE, uPOE] = get_poe(U_t, N_p, Pi)
    dim_n   = size(U_t,2) ;
    U_p     = zeros(size(U_t)) ;
    for n = 1:dim_n       
        U_p(:,n) = N_p*U_t(:,n) ;
    end    
    uPOE = sum((U_t-U_p).^2,2) / dim_n ; 
    vPOE = var(Pi, 0, 2);   
    nPOE = sum(uPOE) / sum(vPOE);
end
