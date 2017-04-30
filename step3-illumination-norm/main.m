clc;
clear;
close all;
rng default;

addpath C:\sdk\M\INface_tool;
addpath C:\sdk\M\INface_tool\auxilary;
addpath C:\sdk\M\INface_tool\mex;
addpath C:\sdk\M\INface_tool\histograms;
addpath C:\sdk\M\INface_tool\photometric;
addpath C:\sdk\M\INface_tool\postprocessors;
addpath C:\sdk\M\INface_tool\demos;


% 枚举文件
fns1 = dir_recursive('../DATA-CROP-RAW/', '*.png');
fns2 = dir_recursive('../DATA-CROP-SYM/', '*.png');
fns = {fns1{:} fns2{:}};
clear fns1 fns2;
assert(numel(fns)==(13202+(245+241+258+262+259)+(865+400+333))*2);

% 随机排序文件列表 防止两个数据库图像大小不同 导致parfor任务量不平衡
fns = fns(randperm(numel(fns)));
assert(numel(fns)==(13202+(245+241+258+262+259)+(865+400+333))*2);

% 修改并发数 可大于或小于线程数
cluster=parcluster;
cluster.NumWorkers=8;
% 并行处理文件
parfor i=1:numel(fns)
    disp([num2str(i) ' / ' num2str(numel(fns))]);
    main_sub1(fns{i});
    main_sub2(fns{i});
end
