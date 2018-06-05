function alpha = ccl_math_uvec (Theta)
% alpha = ccl_math_uvec (Theta)
%
% Convert constraint parameters to constraint vectors e.g. for 3D problem
% a1 = cos(theta1)
% a2 = sin(theta1)*cos(theta2)
% a3 = sin(theta1)*sin(theta2)
% a = [a1,a2,a3]'
%
% Input:
%   Theta                                 Learnt constraint parameters
%
% Output:
%   alpha                                 A unit vector of constraint basis

[dim_n dim_t]   = size(Theta) ;
alpha           = zeros(dim_n,dim_t+1) ;
alpha(:,1)      = cos(Theta(:,1)) ;
for i =2:dim_t
    alpha(:,i) = cos(Theta(:,i)) ;
    
    for k = 1:i-1
        alpha(:,i) = alpha(:,i) .* sin(Theta(:,k)) ;
    end
end
alpha(:,dim_t+1)    = ones(dim_n,1) ;
for k = 1:dim_t
    alpha(:,dim_t+1) = alpha(:,dim_t+1) .* sin(Theta(:,k)) ;
end
end
