This folder contains a sample from the KITTI Stereo 2012 benchmark training set.
(http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)

Contents:
=========

- image_0/1     : grayscale images from the left (image_0) and right (image_1) cameras.
- disp_noc/occ  : sparse ground truth disparity values acquired by accumulating 3D point clouds 
                  from a 360 degree Velodyne HDL-64 Laserscanner. Here the suffix _noc or _occ 
                  refers to whether the disparity is provided only for non-occluded pixels 
                  (i.e., pixels visible to both cameras) or occluded pixels (i.e., all pixels).
                  The disparity images are aligned with the left camera.
- calib         : text files containing 3x4 projection matrices for each of the four KITTI
                  cameras (P0 corresponds to image_0, etc.).
- colored_0/1   : RGB images from the from the left (image_0) and right (image_1) cameras.

File naming examples:
=====================

Stereo pair '000010':
 - left image               : image_0/000010_10.png
 - right image              : image_1/000010_10.png
 - ground truth disparity   : disp_noc/000010_10.png

Data format:
============

Disparity values range [0..255]. Disparity maps are saved as uint16 PNG images, 
which can be opened with MATLAB, libpng++, or Pillow. A 0-value indicates that no ground 
truth exists for that pixel. Otherwise the disparity for a pixel can be computed by
converting the uint16 value to float and dividing it by 256:

disp(u,v)  = ((float)I(u,v))/256.0;
valid(u,v) = I(u,v)>0;
