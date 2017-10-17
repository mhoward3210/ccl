file_names = dir('2017*.mat'); % get data from real robot (previously converted from bag file to *.mat file)
NDem = length(file_names); % number of demonstration
q = cell(NDem,1); % store configuration
u = cell(NDem,1); % store input (configuration velocity)
t = cell(NDem,1);
% Loop to extract and save data
for i=1:NDem
    file_name = file_names(i).name;
    fprintf(1,'Processing file %i: %s ...\n',i,file_name);
    load(file_name); % get data from 1 demonstration
    q_i = cell2mat(data_struct.jointPosition); % joint angles
    dt_i = data_struct.dT(2:end);
    q{i} = num2cell(q_i(2:end,:).',1); % save state: joint angles
    u{i} = num2cell((diff(q_i)./dt_i).',1); % save input: joint velocities
    t{i} = num2cell(cumsum(dt_i).',1); % demonstration time
    % cut initial samples
    N = 100;
    q{i} = q{1}(N:end); u{i} = u{1}(N:end); t{i} = t{1}(N:end); 
end
save('data.mat','q','u','t');