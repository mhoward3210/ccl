function [nmse v mse] = ccl_error_nmse(Y,Yp)
% [nmse v mse] = ccl_error_nmse(Y,Yp)
% 
% Calculate normalised mean square error
%
% Input:
%   
%   Y                      Target data points
%   Yp                     Predictions
%
% Output:
% 
%   nmse                   Normalised mean square error
%   v                      Variance in the observations
%   mse                    Mean square error

% CCL: A MATLAB library for Constraint Consistent Learning
% Copyright (C) 2007  Matthew Howard
% Contact: matthew.j.howard@kcl.ac.uk
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Library General Public
% License along with this library; if not, write to the Free
% Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

N    = size(Y,2);          % get no. data points
mse  = sum((Y-Yp).^2,2)/N; % get mean squared error
v    = var(Y,0,2);         % get variance
nmse = mse/v;              % compute nmse

