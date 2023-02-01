% This demo is provided for image preprocessing of SMU-Net.
% You can easily get the shape constraint map.
% You are free to use, change or redistribute this code for any
% non-commrecial purposes.If you use this software,please cite the
% following in any resulting publication:
% [1] Ning, Zhenyuan and Zhong, Shengzhou and Feng, Qianjin and Chen, Wufan and Zhang, Yu, 
%     SMU-Net: Saliency-Guided Morphology-Aware U-Net for Breast Lesion Segmentation in Ultrasound Image,
%     IEEE Transactions on Medical Imaging,2021


clear all;clc;close all;
fprintf('generating shape groundtruth\n')
file_path = './gt/';% 图像文件夹路径 ?
img_path_list = dir(strcat(file_path,'*.png'));%获取该文件夹中所有PNG格式的图像 ?
img_num = length(img_path_list);%获取图像总数量?
shape_path = './process/scm/';
mkdir(shape_path)

if img_num > 0 
    for m = 1:img_num 
        image_name = img_path_list(m).name;
        I = imread(strcat(file_path,image_name)); 
        fprintf('%d %s\n',m,strcat(file_path,image_name));
        [cloumn,row] = size(I);
        shape = zeros(size(I));
        c = im2bw(I,graythresh(I));
        b = edge(c,'canny');
        [xp,yp] = find(b);   % find edge pixels
        for i = 1:cloumn
            for j = 1:row
                min_dist = 999999;
                for k = 1:size(xp)
                    if i == xp(k) && j == yp(k)
                        min_dist = 0;
                    else
                        if double(sqrt((i - xp(k))^2 + (j - yp(k))^2) + 0.5)< min_dist
                        min_dist = double(sqrt((i - xp(k))^2 + (j - yp(k))^2) + 0.5);%calculate minimal distance
                        end
                    end
                end
                shape(i,j) = min_dist; % assign the distance to corresponding pixel
            end
        end
        shape = 255-uint8(Normalize(shape));
        imwrite(shape,[shape_path,image_name],'png');
    end 
end
