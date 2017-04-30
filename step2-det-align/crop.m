function [im] = crop(im, det, IM_WIDTH, IM_HEIGHT)
    fixedPoints  = [det(1) det(2)
        det(3) det(4)];
    movingPoints = [0 0
        IM_WIDTH IM_HEIGHT];
    im = imwarp(im, fitgeotrans(fixedPoints, movingPoints, 'nonreflectivesimilarity'), 'cubic', 'OutputView', imref2d([IM_WIDTH IM_HEIGHT]));
end
