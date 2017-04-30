clc;
clear;
close all;
rng default;

load bagName;

% 获取所有文件名
fns = dir_recursive('../DATA/SFEW2/', '*');
% fns'

% 分析文件名
TBL = zeros(5,7)
for i=1:numel(fns)
    % 从文件名中分析标签名、电影名
    [lblStr, movie, ~] = fileparts(fns{i});
    [~, lblStr, ~] = fileparts(lblStr);
    movie(movie=='_') = ' ';
    movie = sscanf(movie, '%s', 1);
    movie = lower(movie);
    
    % 提取标签1-7
    % 无标签则为0
    lblStrs = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
    lbl = 0;
    for j=1:7
        if strcmp(lblStr,lblStrs{j})
            assert(lbl==0);
            lbl = j;
        end
    end
    assert(lbl>=1 && lbl<=7)
    % disp([lblStr '	' num2str(lbl)  '	' movie])
    
    % 计算属于哪个cv set
    cvSetId = 0;
    for j=1:5
        for k=1:numel(bagName{j})
            if strcmp(bagName{j}{k}, movie)
                assert(cvSetId==0);
                cvSetId=j;
            end
        end
    end
    assert(cvSetId>=1)
    assert(cvSetId<=5)
    
    % 计算复制目标文件名
    [fnPart1, fnPart4, fnPart5] = fileparts(fns{i});
    [fnPart1, fnPart3, ~] = fileparts(fnPart1);
    [fnPart1, ~, ~] = fileparts(fnPart1);
    fnPart2 = ['Cv' num2str(cvSetId)];
    fn2 = [fnPart1 '/' fnPart2 '/' fnPart3 '/' fnPart4 fnPart5];
    disp(['   ' fns{i}])
    disp(['-> ' fn2])
    
    % 复制
    [dir2, ~, ~] = fileparts(fn2);
    orig_state = warning('off');
    mkdir(dir2)
    warning(orig_state);
    copyfile(fns{i}, fn2)
    
    % 统计分布表
    TBL(cvSetId, lbl) = TBL(cvSetId, lbl)+1;
end

% 输出分布表
TBL(end+1,:) = sum(TBL,1);
TBL(:, end+1) = sum(TBL,2);
disp(TBL)
