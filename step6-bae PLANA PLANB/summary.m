clc;
clear;
close all;

LOG_BASE_DIR = './train_val/';

diary off;
delete([LOG_BASE_DIR 'summary.txt']);
diary([LOG_BASE_DIR 'summary.txt']);

lst = dir([LOG_BASE_DIR '*.txt']);
for i=1:numel(lst)
    if(strcmp(lst(i).name, 'summary.txt'))
        continue;
    end
    fn = [LOG_BASE_DIR lst(i).name];
    summary_sub(fn)
end

diary off;
