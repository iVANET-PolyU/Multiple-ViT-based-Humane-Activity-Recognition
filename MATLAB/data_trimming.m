clc;
clear;
[FileName,PathName,FilterIndex] = uigetfile('*.dat','MultiSelect','on');

for z = 1:length(FileName)
    csi_trace = read_bf_file(strcat(PathName,char(FileName(z))));
    fprintf('package number is %u\n', length(csi_trace));
    
    for i = 1 : fix((length(csi_trace)-600)/300)
        for j = ((i-1)*300+601) : (i*300+600)
            csi_entry = csi_trace{j};
            csi_total = get_scaled_csi(csi_entry);  %2*3*30
            csi = csi_total(1,:,:);   %3*30
            csi_a = abs(squeeze(csi)');
            csi_a = squeeze(csi_a(:));
            for s = 1 : 90
                csi_end(j-((i-1)*300+600),s) = csi_a(s);
            end
        end
        

        csv_base_name = char(FileName(z));
        csv_base_name = csv_base_name(1:end-4);
        if i < 10
            temp = '_00';
            num = num2str(i);
            csv_index = [temp, num];
        elseif i >= 10 && i < 100
            temp = '_0';
            num = num2str(i);
            csv_index = [temp, num];
        elseif i >= 100
            temp = '_';
            num = num2str(i);
            csv_index = [temp, num];
        end
        csv_name = append(csv_base_name, csv_index);
        
        writematrix(csi_end, ['C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\GPU_server\MTL_csitool\Data\test\DWT\CSV_file\',csv_name,'.csv']);
    end
end