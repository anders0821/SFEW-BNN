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

% ö���ļ�
fns = dir_recursive('../DATA/', '*');
assert(numel(fns)==1394+1394+372+13233);

% ��������ļ��б� ��ֹ�������ݿ�ͼ���С��ͬ ����parfor��������ƽ��
fns = fns(randperm(numel(fns)));
assert(numel(fns)==1394+1394+372+13233);

% �޸Ĳ����� �ɴ��ڻ�С���߳���
cluster=parcluster;
cluster.NumWorkers=8;
% ���д����ļ�
parfor i=1:numel(fns)
    disp([num2str(i) ' / ' num2str(numel(fns))]);
    main_sub(fns{i});
end

diary off;
