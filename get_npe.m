%{
calculate the nullspace projection error. This quantity is used to evaluate
the quality of the learnt model in the absence of the true nullspace components

input
    U: true observations
    Unp: learnt nullspace component    
%}
function [umse, variance, nmse] = get_npe (U, Unp)

    N   = size(U,2);
    umse= 0;

    for n=1:N
        up      = Unp(:,n);
        P       = up*up'/(up'*up);
        Pu      = P*U(:,n);
        umse    = umse + norm(Pu - up)^2;
    end

    umse    = umse / N ;
    variance= sum(var(U,1,2)) ;
    nmse    = umse / variance  ;
end
