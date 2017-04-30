function [] = summary_sub(fn)
    f = fopen(fn);
    epoch = nan;
    record = [];
    conMat = {};
    while(true)
        line = fgetl(f);
        if(feof(f))
            break;
        end
        assert(~str_begin_with(line, 'Traceback'));
        
        if(str_begin_with(line, 'Epoch '))
            epoch = sscanf(line, 'Epoch %f');
        end
        if(str_begin_with(line, '  LR:                            '))
            record(epoch).lr = sscanf(line, '  LR:                            %f');
        end
        if(str_begin_with(line, '  training loss:                 '))
            record(epoch).train_loss = sscanf(line, '  training loss:                 %f');
        end
        if(str_begin_with(line, '  validation loss:               '))
            record(epoch).val_loss = sscanf(line, '  validation loss:               %f');
        end
        if(str_begin_with(line, '  training penalty:              '))
            record(epoch).train_penalty = sscanf(line, '  training penalty:              %f');
        end
        if(str_begin_with(line, '  validation penalty:            '))
            record(epoch).val_penalty = sscanf(line, '  validation penalty:            %f');
        end
        if(str_begin_with(line, '('))
            conMat{end+1} = sscanf(line, '(%f,%f)');
        end
    end
    assert(strcmp(line, 'TRIAL END'));
    fclose(f);
    
    conMat = conMat(3:end);
    conMat = cell2mat(conMat);
    % conMat = reshape(conMat, 16 , []);% for group predict
    conMat = reshape(conMat, 8 , []);
    conMat = conMat';
    assert(size(conMat,1)==numel(record))
    
    % ???? ????acc?????
    acc0Train = conMat(:,1) / 0.9;
    acc1Train = conMat(:,4) / 0.1;
    acc01Train = (acc0Train+acc1Train) / 2;
    acc0Val = conMat(:,5) / 0.9;
    acc1Val = conMat(:,8) / 0.1;
    acc01Val = (acc0Val+acc1Val) / 2;
    [~, bestEpoch] = max(acc01Val);
    disp(fn)
    fprintf('%f	%f\n', acc01Train(bestEpoch), acc01Val(bestEpoch));
    
    % ??loss?????
    % [~, bestEpoch] = min([record.val_loss]);
    
    % ??
    close all;
    figure;
    set(gcf,'Position', get(0,'ScreenSize'));
    set(gcf,'Name',fn)
    
    subplot(1,3,1);
    plot([[record.train_loss]' [record.val_loss]' [record.val_penalty]' [record.val_penalty]']);
    hold on
    tmp = [record.val_loss];
    scatter(bestEpoch, tmp(bestEpoch));
    text(bestEpoch, tmp(bestEpoch), ['(' num2str(bestEpoch) ', ' num2str(tmp(bestEpoch)) ')']);
    hold off
    title('loss');
    legend('train loss','val loss', 'train penalty', 'val penalty');
    
    % conMatTitles = {'train', 'train - group predict', 'val', 'val - group predict'};% for group predict
    % conMatSubplotPos = {2, 5, 3, 6};% for group predict
    conMat2 = conMat;
    conMatTitles = {'train', 'val'};
    conMatSubplotPos = {2, 3};
    for i=1:numel(conMatTitles)
        subplot(1,3,conMatSubplotPos{i});
        plot([conMat2(:,1:4) conMat2(:,1)+conMat2(:,2) conMat2(:,1)+conMat2(:,3)]);
        hold on
        scatter([bestEpoch bestEpoch bestEpoch bestEpoch], conMat2(bestEpoch,1:4));
        text(bestEpoch, conMat2(bestEpoch,1), ['(' num2str(bestEpoch) ', ' num2str(conMat2(bestEpoch,1)) ')']);
        text(bestEpoch, conMat2(bestEpoch,2), ['(' num2str(bestEpoch) ', ' num2str(conMat2(bestEpoch,2)) ')']);
        text(bestEpoch, conMat2(bestEpoch,3), ['(' num2str(bestEpoch) ', ' num2str(conMat2(bestEpoch,3)) ')']);
        text(bestEpoch, conMat2(bestEpoch,4), ['(' num2str(bestEpoch) ', ' num2str(conMat2(bestEpoch,4)) ')']);
        hold off
        ylim([0 1]);
        title(conMatTitles{i});
        legend('0->0','0->1','1->0','1->1','input sparsity','reconstructed sparsity');
        conMat2 = conMat2(:, 5:end);
    end
    
    drawnow;
    
    save_current_fig_as_pdf([fn '.pdf'])
    
end
