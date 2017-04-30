function [] = main_sub(fn)
    VIDEO_RESCALE = 2;
    IM_WIDTH = 125;
    IM_HEIGHT = 160;
    
    % 计算输出文件名
    [midfn1, midfn2, ~] = fileparts(fn);
    midfn1 = midfn1(9:end);
    fn2 = ['../DATA-CROP-DET/' midfn1 '/', midfn2 '.png'];
    fn3 = ['../DATA-CROP-SYM/' midfn1 '/', midfn2 '.png'];
    fn4 = ['../DATA-CROP-RAW/' midfn1 '/', midfn2 '.png'];
    [dir2,~,~] = fileparts(fn2);
    [dir3,~,~] = fileparts(fn3);
    [dir4,~,~] = fileparts(fn4);  
    orig_state = warning('off');
    mkdir(dir2);
    mkdir(dir3);
    mkdir(dir4);
    warning(orig_state);
    disp(['   ' fn]);
    disp(['-> ' fn2]);
    disp(['-> ' fn3]);
    disp(['-> ' fn4]);
    
    % 加载图像
    im = imread(fn);
    % 修正视频尺寸 放大计算更精确 检测小目标
    if(all(size(im)==[576 720 3]))
        im = imresize(im, [576*VIDEO_RESCALE 1024*VIDEO_RESCALE]);
        assert(all(size(im)==[576*VIDEO_RESCALE 1024*VIDEO_RESCALE 3]));
    elseif(all(size(im)==[250 250 3]))
        im = imresize(im, [250*VIDEO_RESCALE 250*VIDEO_RESCALE]);
        assert(all(size(im)==[250*VIDEO_RESCALE 250*VIDEO_RESCALE 3]));
    else
        assert(false);
    end
    
    % 特征点定位
    % 失败则尝试 将原始图像镜像 之后处理均以镜像后图像为准
    [dets, fids] = dliblandmarkdetector.detect(im, 0);% dliblandmarkdetector多线程不安全 但parfor是独立进程不受影响
    needMirrorBackOnSave = false;
    if(size(dets,2)==0)
        im = im(:, end:-1:1, :);
        [dets, fids] = dliblandmarkdetector.detect(im, 0);% dliblandmarkdetector多线程不安全 但parfor是独立进程不受影响
        needMirrorBackOnSave = true;
    end
    
    % 延拓dets
    extRatio = 1;
    extDets = dets;
    w = extDets(3,:)-extDets(1,:);
    h = extDets(4,:)-extDets(2,:);
    extH = h.*extRatio./2.0;
    extW = w.*extRatio./2.0;
    extDets(1,:) = extDets(1,:)-extW;
    extDets(2,:) = extDets(2,:)-extH;
    extDets(3,:) = extDets(3,:)+extW;
    extDets(4,:) = extDets(4,:)+extH;
    
    % 显示
    subplot(2,2,1);
    imshow(im);
    hold on;
    plot(fids(1:2:end,:), fids(2:2:end,:))
    for j=1:size(dets,2)
        rectangle('Position', [dets([1 2],j)
            dets([3 4],j)-dets([1 2],j)], 'EdgeColor', 'w');
    end
    for j=1:size(extDets,2)
        rectangle('Position', [extDets([1 2],j)
            extDets([3 4],j)-extDets([1 2],j)], 'EdgeColor', 'g');
    end
    hold off;
    
    % 保存检测结果
    %save([], 'fids', 'dets', 'extDets', 'lbl');
    clear dets
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 0个脸部则跳出
    faceCount = size(extDets,2);
    if(faceCount==0)
        disp(['detection failed: ' fn]);
        
        subplot(2,2,2);
        imshow(zeros([IM_HEIGHT, IM_HEIGHT, 3]));
        subplot(2,2,3);
        imshow(zeros([IM_HEIGHT, IM_WIDTH, 3]));
        subplot(2,2,4);
        imshow(zeros([IM_HEIGHT, IM_WIDTH, 3]));
        drawnow;
        
        %imwrite(zeros([IM_HEIGHT, IM_HEIGHT, 3]), fn2);
        %imwrite(zeros([IM_HEIGHT, IM_WIDTH, 3]), fn3);
        %imwrite(zeros([IM_HEIGHT, IM_WIDTH, 3]), fn4);
        
        % 写空白txt
        [tmp1, tmp2, ~] = fileparts(fn2);
        fn2 = [tmp1 '/' tmp2 '.txt'];
        [tmp1, tmp2, ~] = fileparts(fn3);
        fn3 = [tmp1 '/' tmp2 '.txt'];
        [tmp1, tmp2, ~] = fileparts(fn4);
        fn4 = [tmp1 '/' tmp2 '.txt'];
        fclose(fopen(fn2, 'w'));
        fclose(fopen(fn3, 'w'));
        fclose(fopen(fn4, 'w'));
        return;
    end
    
    % >2个脸部则选最大脸
    if(faceCount>1)
        areas = (extDets(3,:)-extDets(1,:)) .* (extDets(4,:)-extDets(2,:));
        [~, areasMaxIdx] = max(areas);
        extDets = extDets(:,areasMaxIdx);
        fids = fids(:,areasMaxIdx);
    end
    
    % crop
    imCrop = crop(im, extDets, IM_HEIGHT, IM_HEIGHT);
    
    % 扩大DETS裁剪后的 局部FIDS坐标
    fidsCrop = fids;
    fidsCrop(1:2:end) = (fidsCrop(1:2:end)-extDets(1)) * IM_HEIGHT / (extDets(3)-extDets(1));
    fidsCrop(2:2:end) = (fidsCrop(2:2:end)-extDets(2)) * IM_HEIGHT / (extDets(4)-extDets(2));
    
    % 显示
    subplot(2,2,2);
    imshow(imCrop);
    hold on;
    plot(fidsCrop(1:2:end,:), fidsCrop(2:2:end,:))
    hold off;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 正面化
    [frontal_sym, frontal_raw] = face_frontalize(imCrop, fidsCrop);
    
    % 切除方形两侧
    t = ceil((IM_HEIGHT-IM_WIDTH)/2);
    frontal_sym = frontal_sym(:,t:end-t,:);
    frontal_raw = frontal_raw(:,t:end-t,:);
    assert(all(size(frontal_sym)==[IM_HEIGHT, IM_WIDTH, 3]))
    assert(all(size(frontal_raw)==[IM_HEIGHT, IM_WIDTH, 3]))
    
    % 显示
    subplot(2,2,3);
    imshow(frontal_sym);
    subplot(2,2,4);
    imshow(frontal_raw);
    drawnow;
    
    % 保存图像
    if needMirrorBackOnSave
        imwrite(imCrop(:,end:-1:1,:), fn2);
        imwrite(frontal_sym(:,end:-1:1,:), fn3);
        imwrite(frontal_raw(:,end:-1:1,:), fn4);
    else
        imwrite(imCrop, fn2);
        imwrite(frontal_sym, fn3);
        imwrite(frontal_raw, fn4);
    end
end
