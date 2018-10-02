function G = ccl_math_mgmat(dim, i, j, theta)
% G = ccl_math_mgmat(dim, i, j, theta)
% Generate rotation matrix of a plane rotation of degree theta in an arbitrary plane and dimension R
%
% Input:
%   dim                                    Dimensionality of the rotation matrix
%   i,j                                    Row and Col index of the matrix
%   theta                                  Degree of rotation
% Output:
%   G                                      Rotation matrix

G      = eye(dim) ;
G(i,i) = cos(theta) ;
G(j,j) = cos(theta) ;
G(i,j) =-sin(theta) ;
G(j,i) = sin(theta) ;
end
