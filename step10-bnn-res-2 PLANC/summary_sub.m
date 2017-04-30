function [] = summary_sub(fn)
    f = fopen(fn);
    epoch = nan;
    record = [];
    while(true)
        line = fgetl(f);
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
        if(str_begin_with(line, '  training acc:                  '))
            record(epoch).train_acc = sscanf(line, '  training acc:                  %f');
        end
        if(str_begin_with(line, '  validation acc:                '))
            record(epoch).val_acc = sscanf(line, '  validation acc:                %f');
        end
        if(str_begin_with(line, '  validation fuseacc:            '))
            record(epoch).val_fuseacc = sscanf(line, '  validation fuseacc:            %f');
        end
        
        if(feof(f))
            break;
        end
    end
    assert(str_begin_with(line, '  validation fuseacc:            '));
    fclose(f);
    
    % best
    [~, bestEpoch] = max([record.val_acc]);
    
    % ??
    close all;
    figure;
    set(gcf,'Position', get(0,'ScreenSize'));
    set(gcf,'Name',fn)
    
    subplot(2,2,1);
    plot([[record.train_loss]' [record.val_loss]' [record.val_penalty]' [record.val_penalty]']);
    hold on
    tmp = [record.val_loss];
    scatter(bestEpoch, tmp(bestEpoch));
    text(bestEpoch, tmp(bestEpoch), ['(' num2str(bestEpoch) ', ' num2str(tmp(bestEpoch)) ')']);
    hold off
    title('loss');
    legend('train loss','val loss', 'train penalty', 'val penalty');
    grid on;
    
    subplot(2,2,2);
    plot([[record.train_acc]' [record.val_acc]']);
    hold on
    tmp = [record.val_acc];
    scatter(bestEpoch, tmp(bestEpoch));
    text(bestEpoch, tmp(bestEpoch), ['(' num2str(bestEpoch) ', ' num2str(tmp(bestEpoch)) ')']);
    hold off
    title('acc');
    legend('train acc','val acc');
    ylim([0 1])
    grid on;
    
    subplot(2,2,3);
    plot([record.val_fuseacc]');
    hold on
    tmp = [record.val_fuseacc];
    scatter(bestEpoch, tmp(bestEpoch));
    text(bestEpoch, tmp(bestEpoch), ['(' num2str(bestEpoch) ', ' num2str(tmp(bestEpoch)) ')']);
    hold off
    title('acc');
    legend('val fuseacc');
    ylim([0 1])
    grid on;
    
    subplot(2,2,4);
    plot([smooth([record.train_acc]', 100) smooth([record.val_acc]', 100) smooth([record.val_fuseacc]', 100)]);
    title('smooth acc');
    legend('train acc', 'val acc', 'val fuseacc');
    ylim([0 1])
    grid on;
    set(gca,'ytick',0:0.05:1);
    
    drawnow;
    
    save_current_fig_as_pdf([fn '.pdf'])
end
