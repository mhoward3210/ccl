% BAG2MATLAB.m - Extracts the data from all rosbag files in the
% files_directory and stores it as a MatLab structure in a .mat file.
%
% Libraries required: bag2matlab library:
%   https://github.com/unl-nimbus-lab/bag2matlab
%   directory of the library has to be specified using the 
%   bag2matlab_library_path variable
% Subfunctions: bagInfo(), bagReader()

% Author: Joao Moura
% Edinburgh Centre for Robotics, Edinburgh, UK
% email address: Joao.Moura@ed.ac.uk
% Website: http://www.edinburgh-robotics.org/students/joao-moura
% October 2017; Last revision: 17-Oct-2017

%% User Input
bag2matlab_library_path = '../bag2matlab';
files_directory = './';

%% Add bag2matlab library
addpath(bag2matlab_library_path);

%% Get rosbag file names:
file_names = dir(strcat(files_directory,'*.bag'));
if(isempty(file_names)) % test if specified directory has *.bag files
    error(strcat('Error: No bag files found in ',files_directory));
end
N_files = length(file_names); % number of bag files in files_directory

%% Loop to extract and save data
for i=1:N_files % loop over all files_directory ROS bag files
    name = file_names(i).name;
    bag_file = strcat(files_directory,name);
    topics = bagInfo(bag_file); % bagInfo: extract topics info
    if(isempty(topics)) % test if there are any topics
        warning(strcat('Warning: ROS bag file ',name,' has no topic!'));
        continue;
    end
    N_topics = length(topics);
    for n=1:N_topics % loop over all bag file topics
        topic = topics{1,n}; % name of topic
        disp(strcat({'Extracting topic '},topic,{' from  '},name,':'));
        data_table = bagReader(bag_file, topic); % bagReader: get data in a table format
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