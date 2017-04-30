clc;
clear;
close all;

% run('C:/sdk/M/vlfeat-0.9.20/toolbox/vl_setup');

% % ��ȡ�����Ӽ�cv train val stride=1������
% for DLBP_STRIDE = [1]
%     for DLBP_R = 1:15% �뾶17 5��ֲ�����
%         for RAWSYM = {'RAW', 'SYM'}
%             for IN = {'', '-IN-DCT', '-IN-IS'}
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Train/'], ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2TRAIN.mat'], DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Val/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2VAL.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv1/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV1.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv2/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV2.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv3/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV3.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv4/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV4.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv5/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV5.mat']  , DLBP_R, DLBP_STRIDE);
%             end
%         end
%     end
% end

% % ������ȡ�����Ӽ���cv train val stride>1������
% for DLBP_STRIDE = 2:10
%     for DLBP_R = [5]
%         for RAWSYM = {'RAW'}
%             for IN = {''}
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Train/'], ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2TRAIN.mat'], DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Val/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2VAL.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv1/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV1.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv2/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV2.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv3/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV3.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv4/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV4.mat']  , DLBP_R, DLBP_STRIDE);
%                 main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Cv5/']  , ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2CV5.mat']  , DLBP_R, DLBP_STRIDE);
%             end
%         end
%     end
% end

% ������ȡ�����Ӽ�lfw������
% ������ȡ�����Ӽ�test������
for DLBP_STRIDE = [1 2 3]
    for DLBP_R = [5]
        for RAWSYM = {'RAW'}
            for IN = {''}
                main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/lfw/'], ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-LFW.mat'], DLBP_R, DLBP_STRIDE);
                main_sub(['../DATA-CROP-' RAWSYM{1} IN{1} '/SFEW2/Test/'], ['../DATA-CROP-' RAWSYM{1} IN{1} '-F' num2str(DLBP_R) '8' num2str(DLBP_STRIDE) '-LBL-SFEW2TEST.mat'], DLBP_R, DLBP_STRIDE);
            end
        end
    end
end
