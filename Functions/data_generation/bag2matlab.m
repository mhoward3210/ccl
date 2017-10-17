%% File Information:
%
% Extracts the data from all rosbag files in a given directory and saves
% them as Matlab .mat files

%% Add bag2matlab library
bag2matlab_library_path = '../bag2matlab';
addpath(bag2matlab_library_path);

%% Get rosbag file names:
directory = './';
file_names = dir(strcat(directory,'*.bag'));
if(isempty(file_names)) % test if specified direct
    error(strcat('Error: No bag files found in ',directory));
end
N_files = length(file_names); % number of bag files in directory

%% Loop to extract and save data
for i=1:N_files % loop over all directory ros files
    name = file_names(i).name;
    bag_file = strcat(directory,name);
    topics = bagInfo(bag_file);
    if(isempty(topics)) % test if there are any topics
        warning(strcat('Warning: ROS bag file ',name,' has no topic!'));
        continue;
    end
    N_topics = length(topics);
    for n=1:N_topics % loop over all file topics
        topic = topics{1,n}; % name of topic
        disp(strcat({'Extracting topic '},topic,{' from  '},name,':'));
        data_table = bagReader(bag_file, topic); % get data in a table format
        disp('Convert table to struct');
        data_struct = table2struct(data_table,'ToScalar',true); % convert to struct format
        topic = strrep(topic, '/', '_'); % replace / for _
        mat_file_name = strcat(name(1:end-4),topic,'.mat');
        disp('Saving data to file:');
        save(mat_file_name,'data_struct');
        clear data_table data_struct;
    end
    disp('--------------------------------');
end