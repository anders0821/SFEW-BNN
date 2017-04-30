clc;
clear;
close all;
rng default;

% ��ȡ�����ļ���
fns = dir_recursive('../DATA/SFEW2/', '*');
% fns'

% �����ļ���
movieName = {};
movieHist = [];
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
    
    % ͳ��ֱ��ͼ
    row = 0;
    for j=1:size(movieName,1)
        if strcmp(movieName{j, 1}, movie)
            row = j;
            break;
        end
    end
    if(row==0)
        row = size(movieName,1)+1;
        movieName{row, 1} = movie;
        movieHist(row, 1:7) = zeros(1,7);
    end
    movieHist(row, lbl) = movieHist(row, lbl)+1;
end

% ����������ű���
bestCosdist = 0;
tic
while(true)
    % �������
    rndIdx = randperm(size(movieHist,1));
    movieName = movieName(rndIdx, :);
    movieHist = movieHist(rndIdx, :);
    
    % ���հ�����������ƽ��ԭ�� ��������屳��
    count = sum(movieHist,2);
    count = cumsum(count);
    end1 = count(end)/5*1;
    end2 = count(end)/5*2;
    end3 = count(end)/5*3;
    end4 = count(end)/5*4;
    end5 = count(end)/5*5;
    [~, end1] = min(abs(count-end1));
    [~, end2] = min(abs(count-end2));
    [~, end3] = min(abs(count-end3));
    [~, end4] = min(abs(count-end4));
    [~, end5] = min(abs(count-end5));
    assert(0<end1);
    assert(end1<end2);
    assert(end2<end3);
    assert(end3<end4);
    assert(end4<end5);
    assert(end5==95);
    
    bag1 = movieHist(1:end1, :);
    bag2 = movieHist(end1+1:end2, :);
    bag3 = movieHist(end2+1:end3, :);
    bag4 = movieHist(end3+1:end4, :);
    bag5 = movieHist(end4+1:end5, :);
    bag1Name = movieName(1:end1, :);
    bag2Name = movieName(end1+1:end2, :);
    bag3Name = movieName(end2+1:end3, :);
    bag4Name = movieName(end3+1:end4, :);
    bag5Name = movieName(end4+1:end5, :);
    bagName = {bag1Name, bag2Name, bag3Name, bag4Name, bag5Name};
    
    % ������ڸ�lbl����
    ratio1 = sum(bag1);
    ratio2 = sum(bag2);
    ratio3 = sum(bag3);
    ratio4 = sum(bag4);
    ratio5 = sum(bag5);
    ratio1 = ratio1 / norm(ratio1);
    ratio2 = ratio2 / norm(ratio2);
    ratio3 = ratio3 / norm(ratio3);
    ratio4 = ratio4 / norm(ratio4);
    ratio5 = ratio5 / norm(ratio5);
    
    % ��������cosdist�ľ�ֵ Խ��Խ�� ����������lbl��������һ��
    cosdist = [
        ratio1*ratio2'
        ratio1*ratio3'
        ratio1*ratio4'
        ratio1*ratio5'
        ratio2*ratio3'
        ratio2*ratio4'
        ratio2*ratio5'
        ratio3*ratio4'
        ratio3*ratio5'
        ratio4*ratio5'
        ];
    cosdist = mean(cosdist);
    
    % ѡ�����ŵ�cosdist
    if(cosdist > bestCosdist)
        bestCosdist = cosdist;
        bestRndIdx = rndIdx;
        
        disp('----------');
        toc
        tic
        disp(['bestCosdist: ' num2str(bestCosdist)]);
        save bagName bagName
    end
end
