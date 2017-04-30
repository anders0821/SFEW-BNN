function [F] = my_dsift(im)
    assert(all(size(im)==[160 125 3]));
    assert(isa(im, 'uint8'));
    
    % ��ɫͼ��ת ��ͨ�� single 0-1ͼ��
    im = rgb2gray(im);
    im = single(im)/255;
    
    % �����߶�DSIFT
    extractor = dsift.IterDSiftExtractor();
    extractor.num_scales = 5;
    [F, ~] = extractor.compute(im);
    F = double(F(1:128,:));
    
    % ��֤����ά��
    numf = [(160-18)*(125-18)
        (ceil(160/sqrt(2))-18)*(ceil(125/sqrt(2))-18)
        (ceil(160/sqrt(2)/sqrt(2))-18)*(ceil(125/sqrt(2)/sqrt(2))-18)
        (ceil(160/sqrt(2)/sqrt(2)/sqrt(2))-18)*(ceil(125/sqrt(2)/sqrt(2)/sqrt(2))-18)
        (ceil(160/sqrt(2)/sqrt(2)/sqrt(2)/sqrt(2))-18)*(ceil(125/sqrt(2)/sqrt(2)/sqrt(2)/sqrt(2))-18)];
    assert(all(size(F)==[128 sum(numf)]));
end
