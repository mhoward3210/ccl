% Function for loading data in human-readable format files for LSTD(0)-Q learning problems.
%
% in: 
%    filename - filename
%
% out: 
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
%
function D = load_data(filename)

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

