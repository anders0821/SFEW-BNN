function [] = main_sub(IMG_PATH, MAT_PATH, DLBP_R, DLBP_STRIDE)
    disp('----------');
    disp(MAT_PATH);
    if(exist(MAT_PATH, 'file'))
        disp('skip');
        return;
    end
    tic
    
    numf = [(160-2*DLBP_R) (125-2*DLBP_R)
        (ceil(160/sqrt(2))-2*DLBP_R) (ceil(125/sqrt(2))-2*DLBP_R)
        (ceil(160/sqrt(2)/sqrt(2))-2*DLBP_R) (ceil(125/sqrt(2)/sqrt(2))-2*DLBP_R)
        (ceil(160/sqrt(2)/sqrt(2)/sqrt(2))-2*DLBP_R) (ceil(125/sqrt(2)/sqrt(2)/sqrt(2))-2*DLBP_R)
        (ceil(160/sqrt(2)/sqrt(2)/sqrt(2)/sqrt(2))-2*DLBP_R) (ceil(125/sqrt(2)/sqrt(2)/sqrt(2)/sqrt(2))-2*DLBP_R)];
    numf = floor((numf-1)/DLBP_STRIDE) + 1;
    numf = prod(numf,2);
    numf = sum(numf);
    
    % ö���ļ�
    fns = dir_recursive(IMG_PATH, '*.png');
    
    % ���д����ļ�
    F = zeros(10,numf,2,numel(fns), 'uint8');
    LBL = zeros(2,numel(fns));
    % �޸Ĳ����� �ɴ��ڻ�С���߳���
    cluster=parcluster;
    cluster.NumWorkers=8;
    parfor i=1:numel(fns)
        % disp([num2str(i) ' / ' num2str(numel(fns))]);
        [f1, f2, lbl] = main_sub_sub(fns{i}, DLBP_R, DLBP_STRIDE);
        F(:,:,:,i) = cat(3, f1, f2);
        LBL(:,i) = cat(1, lbl, lbl);
    end
    
    % ά��reshapeΪ����ʽ
    F = reshape(F, size(F,1)*size(F,2), size(F,3)*size(F,4));
    LBL = reshape(LBL, size(LBL,1)*size(LBL,2), 1);
    
    % ����mat
    save(MAT_PATH, 'F', 'LBL');
    
    toc
end
