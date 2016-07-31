%{
    Compute the normalised observation error. To evaluate the fit without the 
    true nullspace policy, an alternative criterion must be used. This indicates 
    the quality of the learnt projection matrix in capturing the image space of 
    the observations. i.e., how well the learnt N satisfy Nu = u

    input: 
        f_proj: state dependent prediction of the projection matrix
        X     : true state
        NS_t  : true observed null-space component        
        
    output
        nPOE: normalised projected observation error
        vPOE: variance of the nullspace policy
        uPOE: projected observation error
%}

function [nPOE, vPOE, uPOE] = get_poe_alpha (f_proj, X, NS_t)
    dim_n    = size(NS_t,2) ;
    NS_p     = zeros(size(NS_t)) ;    
    for n = 1:dim_n       
        NS_p(:,n) = f_proj(X(:,n)) * NS_t(:,n) ;        
    end
    uPOE = mean (sum((NS_t-NS_p).^2,1) ) ; 
    vPOE = var(NS_t, 0, 2);   
    nPOE = uPOE / sum(vPOE);
end
