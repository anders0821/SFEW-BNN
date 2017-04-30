function [frontal_sym, frontal_raw] = face_frontalize(im, fid)
    % 静态加载三个数据
    persistent eyemask;
    persistent Model3D;
    persistent REFSZ REFTFORM
    persistent ALIGNED_MASK;
    if isempty(eyemask)
        disp('face_frontalize init');
        
        load eyemask eyemask;
        load model3Ddlib model_dlib;
        Model3D = model_dlib;
        clear model_dlib;
        load DataAlign2LFWa REFSZ REFTFORM
        
        ALIGNED_MASK = face_frontalize_aligned_mask();
    end
    
    % 正面化
    fidu_XY = reshape(fid,2,68);
    fidu_XY = fidu_XY';
    [C_Q, ~,~,~] = estimateCamera(Model3D, fidu_XY);
    [frontal_sym, frontal_raw] = Frontalize(C_Q, im, Model3D.refU, eyemask);% 224->320
    
    % 剪除背景
    % frontal_sym = frontal_sym.*ALIGNED_MASK;
    % frontal_raw = frontal_raw.*ALIGNED_MASK;
    
    % 对齐到LFW坐标系
    %frontal_sym = imtransform(frontal_sym, REFTFORM, 'XData', [1 REFSZ(2)], 'YData',[1 REFSZ(1)]);% 320->250
    %frontal_raw = imtransform(frontal_raw, REFTFORM, 'XData', [1 REFSZ(2)], 'YData',[1 REFSZ(1)]);% 320->250
    
    % 在所需位置裁剪
    % 恢复为输入尺寸
    t = 80;
    b = 240;
    l = (320-(b-t))/2;
    r = l+(b-t);
    frontal_sym = crop(frontal_sym, [l;t;r;b], size(im,1), size(im,2));
    frontal_raw = crop(frontal_raw, [l;t;r;b], size(im,1), size(im,2));
end
