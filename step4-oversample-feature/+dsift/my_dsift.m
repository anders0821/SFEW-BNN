function [F] = my_dsift(im)
    assert(all(size(im)==[160 125 3]));
    assert(isa(im, 'uint8'));
    
    % 彩色图像转 单通道 single 0-1图像
    im = rgb2gray(im);
    im = single(im)/255;
    
    % 计算多尺度DSIFT
    extractor = dsift.IterDSiftExtractor();
    extractor.num_scales = 5;
    [F, ~] = extractor.compute(im);
    F = double(F(1:128,:));
    
    % 验证特征维度
    numf = [(160-18)*(125-18)
        (ceil(160/sqrt(2))-18)*(ceil(125/sqrt(2))-18)
        (ceil(160/sqrt(2)/sqrt(2))-18)*(ceil(125/sqrt(2)/sqrt(2))-18)
        (ceil(160/sqrt(2)/sqrt(2)/sqrt(2))-18)*(ceil(125/sqrt(2)/sqrt(2)/sqrt(2))-18)
        (ceil(160/sqrt(2)/sqrt(2)/sqrt(2)/sqrt(2))-18)*(ceil(125/sqrt(2)/sqrt(2)/sqrt(2)/sqrt(2))-18)];
    assert(all(size(F)==[128 sum(numf)]));
end
