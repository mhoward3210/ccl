function D = ccl_data_load(filename)
% D = ccl_data_load(filename)
%
% Function for loading data in human-readable format files for LSTD(0)-Q learning problems.
%
% Input: 
%    filename - filename
%
% Output: 
%    D        - data struct, containing
%     .T      - time
%     .X      - states
%     .U      - actions
%     .Y      - observed states
%     .Xn     - states after one time step
%     .Un     - actions after one time step (following policy)
%     .Yn     - observed states after one time step
%     .V      - value function at X 
%     .Q      - Q value function at X,U




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

dimX =dlmread([filename,'.dat'],'=',[1,1,1,1]);
dimY =dlmread([filename,'.dat'],'=',[2,1,2,1]);
dimA1=dlmread([filename,'.dat'],'=',[3,1,3,1]);
data =dlmread([filename,'.dat'],'\t',5,0);
D.N = size(data,1); % get no. data points
i=0;
i=i(end)+1:i(end)+dimX      ; D.X = data(:,i)';
i=i(end)+1:i(end)+dimY      ; D.Y = data(:,i)';
i=i(end)+1:i(end)+dimY      ; D.F = data(:,i)';
i=i(end)+1:i(end)+dimA1*dimX; D.A = reshape(data(:,i)',dimA1,dimX,D.N);
i=i(end)+1:i(end)+dimX^2    ; D.P = reshape(data(:,i)',dimX ,dimX,D.N);
end

