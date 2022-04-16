clear;
clc;
[FileName,PathName,FilterIndex] = uigetfile('*.csv','MultiSelect','on');

for z = 1:length(FileName)
    fprintf('This file is called %s\n', char(FileName(z)));
    filename = char(FileName(z));
    filename = filename(1:end-4);
    M = csvread(strcat(PathName,char(FileName(z))));
    Hall = zeros(300,90);
    subcarrier = zeros(300,1);
    MovVar = zeros(300,90);
    vs = zeros(300,1);
    for i = 1: 90
        subcarrier = M(:, i);
        yd=wden(subcarrier,'heursure','s','one',10,'sym3');
%         vs = movvar(yd,50);
        Hall(:,i) = yd;
%         MovVar(:,i) = vs;
%         Combine = cat(2, Hall, MovVar);
    end
    writematrix(Hall, ['C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\GPU_server\MTL_csitool\Data\test\DWT\new\', filename, '.csv']);
%     writematrix(Combine, ['C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\Data\fall_detection\train\DWTwithVar\', filename, '.csv']);
end