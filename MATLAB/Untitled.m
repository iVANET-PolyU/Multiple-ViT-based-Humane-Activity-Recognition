clear;
clc;
csi_trace = read_bf_file('C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\Data\fall_detection\raw_csi\210919\sitdown_210919_Shamshuipo_034.dat');
pac_nEm=size(csi_trace,1);
subcarrier=zeros(1,300);
Hall=zeros(300,30);
Z=zeros(300,30);
Zk=zeros(300,1);
for j=1:90
    %·分别按照时间顺序导入第j个子载波的90个值    
    for i=201:500
        csi_entry=csi_trace{i};
        csi=get_scaled_csi(csi_entry);
        csi1=squeeze(csi(1,:,:)).';% 30*3 complex

        csiabs=abs(csi1);

        csiabs=squeeze(csiabs(:));
        subcarrier(i-200)=csiabs(j);% %10子载波幅度        
    end

    % %第j个子载波滤波
    yd=wden(subcarrier,'heursure','s','one',10,'sym3');
    vs=movvar(yd,50);
    figure(1);
    subplot(3,1,1);
    plot(subcarrier);
    title('beforeDWT');
    ylabel('Amplitude');
    hold on;
    subplot(3,1,2);
    plot(yd);
    title('afterDWT');
    ylabel('Amplitude');
    hold on;
    subplot(3,1,3);
    plot(vs);
    xlabel('package');
    ylabel('variance');
    title('movvar');
    hold on;
    Hall(:,j)=yd.';
end