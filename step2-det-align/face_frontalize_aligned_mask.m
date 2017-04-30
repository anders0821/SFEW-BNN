function [ALIGNED_MASK] = face_frontalize_aligned_mask()
    ALIGNED_MASK = imread('reference_320_320.png');
    ALIGNED_MASK = ALIGNED_MASK==repmat(ALIGNED_MASK(1,1,:), [320 320 1]);
    ALIGNED_MASK = ~all(ALIGNED_MASK, 3);
    ALIGNED_MASK = repmat(ALIGNED_MASK, [1 1 3]);
    ALIGNED_MASK = uint8(ALIGNED_MASK);
end
