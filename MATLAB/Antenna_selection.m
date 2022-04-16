[FileName,PathName,FilterIndex] = uigetfile('*.dat','MultiSelect','on');

for z = 1:length(FileName)
    csi_trace = read_bf_file(strcat(PathName,char(FileName(z))));
    disp(FileName(z));
    fprintf('package number is %u\n', length(csi_trace));
    if length(csi_trace)>=400
        for i = 101 : 400
            csi_entry = csi_trace{i};
            csi_total = get_scaled_csi(csi_entry);  %2*3*30
            csi = csi_total(1,:,:);   %3*30
            csi_a = abs(squeeze(csi).');
            for j = 1 : 30
                csi_end(i-100,j,:) = csi_a(j,:);
            end
        end
    else
        for i = 51 : 350
            csi_entry = csi_trace{i};
            csi_total = get_scaled_csi(csi_entry);  %2*3*30
            csi = csi_total(1,:,:);   %3*30
            csi_a = abs(squeeze(csi).');
            for j = 1 : 30
                csi_end(i-50,j,:) = csi_a(j,:);
            end
        end
    end
    
    csi_1 = csi_end(:,:,1);
    csi_2 = csi_end(:,:,2);
    csi_3 = csi_end(:,:,3);

    
    W = 40; % 60/1.5, 60 is sampling frequency
    step = 2;
    r_j_max = 0;
    csi_avg_1 = mean(csi_1,2);
    csi_avg_2 = mean(csi_2,2);
    csi_avg_3 = mean(csi_3,2);

    csi_avg = [csi_avg_1,csi_avg_2,csi_avg_3];
    ant_index = 0;
    for j = 1:3 
        n_j = length(csi_avg(:,j));
        e_j = [];
        k = 1;
        while  k+W <  n_j
            v_k = var(csi_avg(k:k+W,j));
            e_j = [e_j, v_k];
            k = k + step;
        end
        r_j = max(e_j) - min(e_j);
        if r_j > r_j_max
            r_j_max = r_j;
            ant_index = j;
        end
    end
    
    if ant_index == 1
        writematrix(csi_1, ['C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\Data\fall_detection\train\Antenna_selection\',char(FileName(z)),'.csv']);
    elseif ant_index == 2
        writematrix(csi_2, ['C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\Data\fall_detection\train\Antenna_selection\',char(FileName(z)),'.csv']);
    else 
        writematrix(csi_3, ['C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\Data\fall_detection\train\Antenna_selection\',char(FileName(z)),'.csv']);
    end
end    


