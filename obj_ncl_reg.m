%{
learning objective function for learning nullspace component. The goal is 
to model the nullspace component Uns such that the difference between P*U 
and the learnt Uns is minimised. P is a projection matrix that projects U 
onto the image space of Uns. 

input
    model: current model
    W: weights
    BX: B(X)  
    U: observed actions
output
    fun: error function of weight W
    J: jacobian
%}

function [fun J] = obj_ncl_reg (model, W, BX, U)   
    [dim_U, dim_N ] = size(U) ;
    lambda  = 1e-8;    
    W       = reshape(W, dim_U, model.num_basis );  
    J       = zeros( dim_N, model.num_basis * dim_U);
    fun     = zeros( dim_N, 1 );
    
    for n = 1 : dim_N
        b_n     = BX(:,n);
        u_n     = U (:,n);
        Wb      = W    * b_n;        
        c       = Wb'  * Wb;
        a       = u_n' * Wb;
        j_n     = ( u_n*b_n'*c - Wb*b_n'*(c+a - 2*lambda) ) / (sqrt(c)*c);
        J(n,:)  = j_n(:);
        fun(n)  = (a-c + lambda)/sqrt(c);        
    end            
end
