function [r] = str_begin_with(str, substr)
    r = numel(str)>=numel(substr) && all(str(1:numel(substr))==substr);
end
