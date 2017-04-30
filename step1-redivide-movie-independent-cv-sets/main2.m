clc;
clear;
close all;
rng default;

load bagName;

% ��ȡ�����ļ���
fns = dir_recursive('../DATA/SFEW2/', '*');
% fns'

% �����ļ���
TBL = zeros(5,7)
for i=1:numel(fns)
    % ���ļ����з�����ǩ������Ӱ��
    [lblStr, movie, ~] = fileparts(fns{i});
    [~, lblStr, ~] = fileparts(lblStr);
    movie(movie=='_') = ' ';
    movie = sscanf(movie, '%s', 1);
    movie = lower(movie);
    
    % ��ȡ��ǩ1-7
    % �ޱ�ǩ��Ϊ0
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
    
    % ���������ĸ�cv set
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
    
    % ���㸴��Ŀ���ļ���
    [fnPart1, fnPart4, fnPart5] = fileparts(fns{i});
    [fnPart1, fnPart3, ~] = fileparts(fnPart1);
    [fnPart1, ~, ~] = fileparts(fnPart1);
    fnPart2 = ['Cv' num2str(cvSetId)];
    fn2 = [fnPart1 '/' fnPart2 '/' fnPart3 '/' fnPart4 fnPart5];
    disp(['   ' fns{i}])
    disp(['-> ' fn2])
    
    % ����
    [dir2, ~, ~] = fileparts(fn2);
    orig_state = warning('off');
    mkdir(dir2)
    warning(orig_state);
    copyfile(fns{i}, fn2)
    
    % ͳ�Ʒֲ���
    TBL(cvSetId, lbl) = TBL(cvSetId, lbl)+1;
end

% ����ֲ���
TBL(end+1,:) = sum(TBL,1);
TBL(:, end+1) = sum(TBL,2);
disp(TBL)
