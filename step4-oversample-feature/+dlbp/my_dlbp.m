function [F] = my_dlbp(im, DLBP_R, DLBP_STRIDE)
    assert(all(size(im)==[160 125 3]));
    assert(isa(im, 'uint8'));
    
    % ��ɫͼ��ת ��ͨ�� uint8 0-255ͼ��
    im = rgb2gray(im);
    
    % �����߶�DLBP
    F = {};
    for i=1:5
        scale = 0.5^((i-1)/2);
        % ע�⣺��ͬ�汾matlab˫���β�ֵimresize�ı߽紦��ʽ��һ�������ܻ�Ӱ��ʵ�鸴�֣���ʵ�������matlab 2014b��2015b
        imScale = imresize(im, ceil(size(im)*scale));
        
        % ���㵥�߶�DSIFT
        map = dlbp.getmapping(8, 'riu2');% 8�ڵ�, ��ת���� uniform
        F{i} = dlbp.lbp(imScale, DLBP_R, 8, map, '');% 1�뾶, 8�ڵ�, ��ת���� uniform, ����ܶ�ͼ
        F{i} = double(F{i});
        
        % ����
        codes = (min(map.table):max(map.table))';
        F{i} = permute(F{i}, [3 1 2]);
        F{i} = bsxfun(@eq, F{i}, codes);
        F{i} = double(F{i});
    end
    
    % stride�������
    for i=1:5
        F{i} = F{i}(:,1:DLBP_STRIDE:end,1:DLBP_STRIDE:end);
    end
    
    % ���Ӽ����߶�DLBP����
    % ����˳����dsiftһ�� (W*Hͼ��1,��2,...) (0.7W*0.7Hͼ��1,��2,...) (0.5W*0.5Hͼ��1,��2...)
    for i=1:5
        F{i} = reshape(F{i}, [size(F{i},1) size(F{i},2)*size(F{i},3)]);
    end
    F = cell2mat(F);
    
    % ��֤����ά��
    numf = [(160-2*DLBP_R) (125-2*DLBP_R)
        (ceil(160/sqrt(2))-2*DLBP_R) (ceil(125/sqrt(2))-2*DLBP_R)
        (ceil(160/sqrt(2)/sqrt(2))-2*DLBP_R) (ceil(125/sqrt(2)/sqrt(2))-2*DLBP_R)
        (ceil(160/sqrt(2)/sqrt(2)/sqrt(2))-2*DLBP_R) (ceil(125/sqrt(2)/sqrt(2)/sqrt(2))-2*DLBP_R)
        (ceil(160/sqrt(2)/sqrt(2)/sqrt(2)/sqrt(2))-2*DLBP_R) (ceil(125/sqrt(2)/sqrt(2)/sqrt(2)/sqrt(2))-2*DLBP_R)];
    numf = floor((numf-1)/DLBP_STRIDE) + 1;
    numf = prod(numf,2);
    numf = sum(numf);
    assert(all(size(F)==[10 numf]));
end
