function [a] = bitrol_double(a,l,k)
    assert(isscalar(a));
    assert(a==floor(a));
    a = fi(a,0,l,0);
    % disp(bin(a))
    a = bitrol(a,k);
    % disp(bin(a))
    
    a = double(a);
end
