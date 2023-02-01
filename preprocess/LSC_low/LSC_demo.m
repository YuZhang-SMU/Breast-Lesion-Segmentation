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
file_path = 'D:\DATA\thyriod_nodule\Fold1\test\ori\';% 图像文件夹路径 ?
img_path_list = dir(strcat(file_path,'*.png'));%获取该文件夹中所有PNG格式的图像 ?
img_num = length(img_path_list);%获取图像总数量?
mkdir 'D:\DATA\thyriod_nodule\Fold1\ori_8';
if img_num > 0 %有满足条件的图像 ?
    for j = 1:img_num %逐一读取图像 ?
        image_name = img_path_list(j).name;% 图像名 ?
        img = imread(strcat(file_path,image_name)); 
        fprintf('%d %s\n',j,strcat(file_path,image_name));% 显示正在处理的图像名 
        %图像处理过程 若图像格式不对，matlab将会强制退出
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





