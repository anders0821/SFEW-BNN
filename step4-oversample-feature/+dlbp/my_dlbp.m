function [F] = my_dlbp(im, DLBP_R, DLBP_STRIDE)
    assert(all(size(im)==[160 125 3]));
    assert(isa(im, 'uint8'));
    
    % 彩色图像转 单通道 uint8 0-255图像
    im = rgb2gray(im);
    
    % 计算多尺度DLBP
    F = {};
    for i=1:5
        scale = 0.5^((i-1)/2);
        % 注意：不同版本matlab双三次插值imresize的边界处理方式不一样，可能会影响实验复现，本实验混用了matlab 2014b与2015b
        imScale = imresize(im, ceil(size(im)*scale));
        
        % 计算单尺度DSIFT
        map = dlbp.getmapping(8, 'riu2');% 8邻点, 旋转不变 uniform
        F{i} = dlbp.lbp(imScale, DLBP_R, 8, map, '');% 1半径, 8邻点, 旋转不变 uniform, 输出密度图
        F{i} = double(F{i});
        
        % 解码
        codes = (min(map.table):max(map.table))';
        F{i} = permute(F{i}, [3 1 2]);
        F{i} = bsxfun(@eq, F{i}, codes);
        F{i} = double(F{i});
    end
    
    % stride间隔采样
    for i=1:5
        F{i} = F{i}(:,1:DLBP_STRIDE:end,1:DLBP_STRIDE:end);
    end
    
    % 连接计算多尺度DLBP特征
    % 连接顺序与dsift一致 (W*H图列1,列2,...) (0.7W*0.7H图列1,列2,...) (0.5W*0.5H图列1,列2...)
    for i=1:5
        F{i} = reshape(F{i}, [size(F{i},1) size(F{i},2)*size(F{i},3)]);
    end
    F = cell2mat(F);
    
    % 验证特征维度
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
