% You are free to use, change or redistribute this code for any
% non-commrecial purposes.If you use this software,please cite the
% following in any resulting publication and email us:
% [1] Zhengqin Li, Jiansheng Chen, Superpixel Segmentation using Linear 
%     Spectral Clustering, IEEE Conference on Computer Vision and Pattern 
%     Recognition (CVPR), Jun. 2015 
% (C) Zhengqin Li, Jiansheng Chen, 2014
% li-zq12@mails.tsinghua.edu.cn
% jschenthu@mail.tsinghua.edu.cn
% Tsinghua University

clear all;clc;close all;
file_path = 'D:\DATA\thyriod_nodule\Fold1\test\ori\';% ͼ���ļ���·�� ?
img_path_list = dir(strcat(file_path,'*.png'));%��ȡ���ļ���������PNG��ʽ��ͼ�� ?
img_num = length(img_path_list);%��ȡͼ��������?
mkdir 'D:\DATA\thyriod_nodule\Fold1\ori_8';
if img_num > 0 %������������ͼ�� ?
    for j = 1:img_num %��һ��ȡͼ�� ?
        image_name = img_path_list(j).name;% ͼ���� ?
        img = imread(strcat(file_path,image_name)); 
        fprintf('%d %s\n',j,strcat(file_path,image_name));% ��ʾ���ڴ����ͼ���� 
        %ͼ������� ��ͼ���ʽ���ԣ�matlab����ǿ���˳�
%         if (length(size(img))<2)
        img=repmat(img,[1,1,3]);
%         end
        gaus=fspecial('gaussian',3);
        I=imfilter(img,gaus);
        superpixelNum=8;
        ratio=0.075;

        label=LSC_mex(I,superpixelNum,ratio);
        label0=uint8(Normalize(label));
%         imshow(label0);
        imwrite(label0,['D:\DATA\thyriod_nodule\Fold1\ori_8\',image_name],'png');
    end 
end 

% name='000002';
% img0=imread([name,'.png']);
% img=repmat(img0,[1,1,3]);
% gaus=fspecial('gaussian',3);
% I=imfilter(img,gaus);
% 
% superpixelNum=30;
% ratio=0.075;
% 
% label=LSC_mex(I,superpixelNum,ratio);
% 
% imshow(label,[]);
% imwrite(label,[name,'label.png'],'png');

% DisplaySuperpixel(label,img,name);

% DisplayLabel(label,name);





