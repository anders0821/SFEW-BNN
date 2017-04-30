function [f1, f2, lbl] = main_sub_sub(fn, DLBP_R, DLBP_STRIDE)
    % 读取图片
    im = imread(fn);
    assert(all(size(im)==[160 125 3]));
    
    % 提取特征
    f1 = dlbp.my_dlbp(im, DLBP_R, DLBP_STRIDE);
    f1 = uint8(f1);
    
    % 提取mirror特征
    f2 = dlbp.my_dlbp(im(:,end:-1:1,:), DLBP_R, DLBP_STRIDE);
    f2 = uint8(f2);
    
    % 提取标签1-7
    % 无标签则为0
    lblStrs = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
    lbl = 0;
    for j=1:7
        if numel(strfind(fn, lblStrs{j}))>0
            assert(lbl==0);
            lbl = j;
        end
    end
    lbl = uint8(lbl);
end
