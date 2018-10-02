function ccl_data_read_save(x,y,z,N,name)
% ccl_data_read_save(x,y,z,N,name)
%
% Data preparation for CCL learning
%
% Input:
%
%   x,y,z                   Real data in column
%       N                   Number of data remain
%       name                Name of the generated file




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

N_ = size(x,1);
data.N = N;
ind = floor((N_-N)/2):(floor((N_-N)/2)+N-1);
data.X = [x';y';z'];
data.Y = [diff(data.X')/0.02]';
data.Y = [reshape(smooth(data.Y',10),size(data.Y,2),size(data.Y,1))]';
data.F = zeros(3,N);
data.A = zeros(1,3,N);
data.P = zeros(3,3,N);
data.X = data.X(:,ind);
data.Y = data.Y(:,ind);
figure;scatter3(data.X(1,:),data.X(2,:),data.X(3,:));
xlabel('x');ylabel('y');zlabel('z');zlim([0,1])
ccl_data_save(name,data);
end
