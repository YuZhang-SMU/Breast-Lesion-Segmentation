% This demo is provided for image preprocessing of SMU-Net.
% You can easily get the position map.
% You are free to use, change or redistribute this code for any
% non-commrecial purposes.If you use this software,please cite the
% following in any resulting publication:
% [1] Ning, Zhenyuan and Zhong, Shengzhou and Feng, Qianjin and Chen, Wufan and Zhang, Yu, 
%     SMU-Net: Saliency-Guided Morphology-Aware U-Net for Breast Lesion Segmentation in Ultrasound Image,
%     IEEE Transactions on Medical Imaging,2021


clear all;clc;close all;
% generate position map
fprintf('generating position map\n')
file_path = './original/';
img_path_list = dir(strcat(file_path,'*.png'));
img_num = length(img_path_list);
pos_size = [32,64,128,256];

if img_num > 0 
    for t = 1:length(pos_size)
        pos_path = strcat('./process/pos/pos',num2str(t),'/');
        mkdir(pos_path);
        q = (256/pos_size(t));
        for j = 1:img_num
            image_name = img_path_list(j).name;
            image = imread(strcat(file_path,image_name));
            fprintf('%d %s\n',j, strcat(file_path,image_name));
            coor=xlsread('./point_original.xls','Sheet1','B1:G10');
            pos_maps = zeros(pos_size(t));
            center_x = (coor(j,1)+coor(j,3)+coor(j,5))/3/q+0.5;
            center_y = (coor(j,2)+coor(j,4)+coor(j,6))/3/q+0.5;
            min_dist = min([center_x,center_y,pos_size(t)-center_y,pos_size(t)-center_x]);
            for x = 1:pos_size(t)
                for y = 1:pos_size(t)
                    distance = sqrt((center_x-x)^2+(center_y-y)^2);
                    if distance > min_dist
                        pos_maps(x,y) = min_dist;
                    else
                        pos_maps(x,y) = distance;
                    end
                end
            end
            pos_maps = uint8(255-Normalize(pos_maps));
            imwrite(pos_maps,[pos_path,image_name],'png');
        end
    end
end