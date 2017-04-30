clc;
clear;
close all;
rng default;

diary off;
delete diary.txt;
diary diary.txt;

addpath C:\sdk\M\dliblandmarkdetector\;
addpath C:\sdk\M\frontalize.0.1.2-mod\;
addpath C:\sdk\M\frontalize.0.1.2-mod\calib\;

% 枚举文件
fns = dir_recursive('../DATA/', '*');
assert(numel(fns)==1394+1394+372+13233);

% 随机排序文件列表 防止两个数据库图像大小不同 导致parfor任务量不平衡
fns = fns(randperm(numel(fns)));
assert(numel(fns)==1394+1394+372+13233);

% 修改并发数 可大于或小于线程数
cluster=parcluster;
cluster.NumWorkers=8;
% 并行处理文件
parfor i=1:numel(fns)
    disp([num2str(i) ' / ' num2str(numel(fns))]);
    main_sub(fns{i});
end

diary off;
