# SMU-Net: Saliency-guided Morphology-aware U-Net for Breast Lesion Segmentation in Ultrasound Image

This is an implementation of SMU-Net. We will introduce how to preprocess data to generate foreground and background saliency map.
Meanwhile, we also provide a keras & pytorch implementation of our model to meet with the need of different users.

![image](https://github.com/YuZhang-SMU/Breast-Lesion-Segmentation/blob/master/readme/SMU-Net.png)

#
image preprocessing includes seed point generation, high-level saliency map generation, final foreground&background saliency map generation, shape constraint map generation and position map generation.

# seed point generation
(1) Run './preprocess/seed_point_generation/dist/seed_point.exe'.

(2) Click the button of 'Open dir' to open image dir.
![image](https://github.com/YuZhang-SMU/Breast-Lesion-Segmentation/blob/master/readme/1.png)

(3). Click the target region of imgae for three times to generate three seed points.
![image](https://github.com/YuZhang-SMU/Breast-Lesion-Segmentation/blob/master/readme/2.png)
![image](https://github.com/YuZhang-SMU/Breast-Lesion-Segmentation/blob/master/readme/3.png)

(4) Click the button of 'Save coordinates' to save their coordinates into the excel files (named as 'point_'+dir_name, i.e., 'point_original.xls').
![image](https://github.com/YuZhang-SMU/Breast-Lesion-Segmentation/blob/master/readme/4.png)

(5) Click the button of 'Next image' and repeat step (2)-(5).
![image](https://github.com/YuZhang-SMU/Breast-Lesion-Segmentation/blob/master/readme/5.png)

(6) If all files are annotated, it will prompt that all files in this folder are finished. You can select the next folder to continue the annotations or exit.
![image](https://github.com/YuZhang-SMU/Breast-Lesion-Segmentation/blob/master/readme/6.png)

Note that, all annotated images will be backed up to the corresponding folder (i.e.,'./original_backup') for quick location of the unannotated images.
If the program terminates due to some unexpected operation, just rerun it and continue your annotations.
If the coordinates are out of the image, do not save!!! 
You can reclick to generate three new seed points and save them to overwrite the wrong coordinates.

# high-level saliency map generation
(1) Unzip './preprocess/mcg_high.zip' in Linux. Note that, you must unzip and run the code in Linux, because the core code of mcg is implemented in Linux.

(2) Run './preprocess/mcg_high/install.m' to configure the paths and parameters.

(3) Run './preprocess/mcg_high/demos/demo_im2mcg.m'

(4) copy the high-level saliency map in the dir of './preprocess/mcg_high/demos/results/sp_high' and save into the dir of './preprocess'

# foreground&background saliency map generation
Run './preprocess/LSC_low/saliency_map.m'.

# shape constraint map generation
Run './preprocess/LSC_low/shape_constraint_map.m'.

# position map generation
Run './preprocess/LSC_low/position_map.m'.

#

Finally, all processed images are saved into the dir of './preprocess/LSC_low/process'.
We have provided some processed image to help follow and understand our code.

The requirements of environment is:
Python 3.6.5
Keras 2.2.4 / torch 1.10.1
tensorflow 1.13.0
Matlab R2017b (Linux) / R2018a (Windows)

If you are interested in this work, please cite this paper: 

Z. Ning, S. Zhong, Q. Feng, W. Chen and Y. Zhang, "SMU-Net: Saliency-guided Morphology-aware U-Net for Breast Lesion Segmentation in Ultrasound Image," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2021.3116087.
