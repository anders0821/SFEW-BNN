function [fns] = dir_recursive(path, filter)
    fns = {};
    
    % �������
    % Ŀ¼
    lst = dir(path);
    for i=1:numel(lst);
        fn = lst(i).name;
        if(strcmp(fn,'.') || strcmp(fn,'..'))
            % . ..
            continue;
        end
        if(isdir([path fn]))
            fns2 = dir_recursive([path fn '/'], filter);
            fns(end+1:end+numel(fns2)) = fns2;
        end
    end
    
    % �ļ�
    lst = dir([path filter]);
    for i=1:numel(lst);
        fn = lst(i).name;
        if(strcmp(fn,'.') || strcmp(fn,'..'))
            % . ..
            continue;
        end
        
        if(~isdir([path fn]))
            fns{end+1} = [path fn];
        end
    end
end
