function [] = train_val_trial(rawsym_in, DLBP_R, DLBP_STRIDE)
    addpath C:\sdk\M\liblinear-2.1\windows;
    
    % 加载数据
    disp('preparing data');
    SFEW2TRAIN = load(['../DATA-CROP-' rawsym_in '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2TRAIN.mat']);
    SFEW2VAL = load(['../DATA-CROP-' rawsym_in '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2VAL.mat']);
    
    % reshuffle
    rng default% 经测试独立job中随机数发生器是独立的
    rndIdx = randperm(numel(SFEW2TRAIN.LBL));
    SFEW2TRAIN.F = SFEW2TRAIN.F(:,rndIdx);
    SFEW2TRAIN.LBL = SFEW2TRAIN.LBL(rndIdx);
    
    % reshuffle
    rng default% 经测试独立job中随机数发生器是独立的
    rndIdx = randperm(numel(SFEW2VAL.LBL));
    SFEW2VAL.F = SFEW2VAL.F(:,rndIdx);
    SFEW2VAL.LBL = SFEW2VAL.LBL(rndIdx);
    
    % 修正数据为liblinear适用的格式
    SFEW2TRAIN.F = SFEW2TRAIN.F';
    SFEW2TRAIN.F = double(SFEW2TRAIN.F);
    SFEW2TRAIN.F = sparse(SFEW2TRAIN.F);
    
    % 修正数据为liblinear适用的格式
    SFEW2VAL.F = SFEW2VAL.F';
    SFEW2VAL.F = double(SFEW2VAL.F);
    SFEW2VAL.F = sparse(SFEW2VAL.F);
    
    % 显示数据大大小
    disp(size(SFEW2TRAIN.F));
    disp(size(SFEW2TRAIN.LBL));
    disp(size(SFEW2VAL.F));
    disp(size(SFEW2VAL.LBL));
    
    % 训练
    disp('training');
    model = train(SFEW2TRAIN.LBL, SFEW2TRAIN.F, '');

    % 验证
    disp('validating');
    [predicted, ~, ~] = predict(SFEW2TRAIN.LBL, SFEW2TRAIN.F, model, '');
    acc_train = sum(predicted==SFEW2TRAIN.LBL) / numel(SFEW2TRAIN.LBL);
    [predicted, ~, ~] = predict(SFEW2VAL.LBL, SFEW2VAL.F, model, '');
    acc_val = sum(predicted==SFEW2VAL.LBL) / numel(SFEW2VAL.LBL);
    
    disp('----------')
    disp([num2str(acc_train*100) '%	'  num2str(acc_val*100) '%'])
    disp(predicted)
end
