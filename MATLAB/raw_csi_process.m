clc;
clear;
[FileName,PathName,FilterIndex] = uigetfile('*.dat','MultiSelect','on');

for z = 1:length(FileName)
    csi_trace = read_bf_file(strcat(PathName,char(FileName(z))));
    fprintf('This file is called %s\n', char(FileName(z)));
    fprintf('package number is %u\n', length(csi_trace));
    
    filename = char(FileName(z));
    filename = filename(1:end-4);
    csi_end = [];
    
    if length(csi_trace)>300
        for i = 1:300
            csi_entry = csi_trace{i};
            csi_total = get_scaled_csi(csi_entry);
            csi = csi_total(1,:,:);  
            csi_a = abs(squeeze(csi).');
            csi_a = squeeze(csi_a(:));
            for l = 1 : 90
                csi_end(i,l) = csi_a(l);
            end   
        end
%     elseif length(csi_trace)>450 && length(csi_trace)<500
%         for i = 151:450
%             csi_entry = csi_trace{i};
%             csi_total = get_scaled_csi(csi_entry);
%             csi = csi_total(1,:,:);  
%             csi_a = abs(squeeze(csi).');
%             csi_a = squeeze(csi_a(:));
%             for l = 1 : 90
%                 csi_end(i-150,l) = csi_a(l);
%             end   
%         end
    end
    
    if isempty(csi_end) == 0
        Hall = zeros(300,90);
        subcarrier = zeros(300,1);
        for i = 1: 90
            subcarrier = csi_end(:, i);
            yd=wden(subcarrier,'heursure','s','one',10,'sym3');
            Hall(:,i) = yd;
        end
        writematrix(Hall, ['C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\GPU_server\MTL_csitool\Data\test\DWT\new\', filename, '.csv']);
    end
    
  
end