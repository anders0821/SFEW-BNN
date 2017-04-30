clc
clear
close all




diary off
delete fns-train-det.txt
delete fns-train-nodet.txt
delete fns-val-det.txt
delete fns-val-nodet.txt
delete fns-test-det.txt
delete fns-test-nodet.txt



diary off
diary fns-train-det.txt
diary on
fns = dir_recursive('../DATA-CROP-RAW/SFEW2/Train/', '*.png');
for fn=fns
    disp(fn{1});
end
diary off

diary off
diary fns-train-nodet.txt
diary on
fns = dir_recursive('../DATA-CROP-RAW/SFEW2/Train/', '*.txt');
for fn=fns
    disp(fn{1});
end
diary off





diary off
diary fns-val-det.txt
diary on
fns = dir_recursive('../DATA-CROP-RAW/SFEW2/Val/', '*.png');
for fn=fns
    disp(fn{1});
end
diary off

diary off
diary fns-val-nodet.txt
diary on
fns = dir_recursive('../DATA-CROP-RAW/SFEW2/Val/', '*.txt');
for fn=fns
    disp(fn{1});
end
diary off





diary off
diary fns-test-det.txt
diary on
fns = dir_recursive('../DATA-CROP-RAW/SFEW2/Test/', '*.png');
for fn=fns
    disp(fn{1});
end
diary off

diary off
diary fns-test-nodet.txt
diary on
fns = dir_recursive('../DATA-CROP-RAW/SFEW2/Test/', '*.txt');
for fn=fns
    disp(fn{1});
end
diary off
