function [] = main_sub2(fn)
    % 计算输出文件名
    assert(strcmp(fn(1:17), '../DATA-CROP-SYM/') || strcmp(fn(1:17), '../DATA-CROP-RAW/'));
    fn2 = [fn(1:16) '-IN-DCT' fn(17:end)];
    [dir2,~,~] = fileparts(fn2);
    orig_state = warning('off');
    mkdir(dir2);
    warning(orig_state);
    disp(['   ' fn]);
    disp(['-> ' fn2]);
    
    % 加载图像
    im = imread(fn);
    
    % rgbh->sv
    hsv = rgb2hsv(im);% 0-1 double
    gray = hsv(:,:,3);
    
    % 转为方图
    %assert(all(size(gray)==[160 125]));
    %gray = imresize(gray, [160 160]);
    
    % IN
    gray = gray*255.0;
    gray = DCT_normalization(gray);% demo中输入图像格式是0-255 double 128*128
    gray = gray/255.0;
    
    % 恢复原尺寸
    %gray = imresize(gray, [160 125]);
    
    % hsv->rgb
    hsv(:,:,3) = gray;
    im = hsv2rgb(hsv);
    
    % 存储图像
    imwrite(im, fn2);
end
