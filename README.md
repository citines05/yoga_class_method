# Lightweight Method for Yoga Posture Recognition

Recognizing the growing importance of yoga for enhancing physical health and mental well-being, this project proposes a lightweight neural network method for the automatic recognition of yoga postures from images. By leveraging skeletal keypoints, our model achieves efficient and accurate posture classification. We evaluated our approach on the Yoga-82 dataset using two data augmentation strategies: horizontal flipping of images and data balancing via random Gaussian noise addition combined with keypoint fusion. Our model attains an accuracy of 90.31\% with only 85,582 parameters for 82 distinct yoga postures, demonstrating competitive performance relative to more resource-intensive methods. This efficiency makes the approach particularly suitable for resource-constrained environments, such as smartphones, and paves the way for developing tutor applications that promote individual yoga practice and enhance overall well-being.

## **Note:** Due to the size of the datasets, they are stored in a folder on [Google Drive](https://drive.google.com/drive/folders/1J22NMrp7-ASANnqbkdPJ8ay9WPHqV_VG?usp=sharing).

## Instructions to recreate each dataset used

### **`kp_ds_not_aug_1`**

- Extract the directory from the zipped file `img_ds_not_clean_not_aug.zip` and execute the following scripts in order:
  1. `img_data_cl.py`
  2. `img_data_redist.py`
  3. `kp_ds_gen.py` (note: change the final directory name to `kp_ds_not_aug_1`)

### **`kp_ds_aug_2`**

- Based on the already cleaned directory, execute the following scripts in order:
  1. `img_data_aug.py`
  2. `kp_ds_gen.py` (note: change the final directory name to `kp_ds_aug_2`)

### **`kp_ds_aug_balan_3`**

- Based on the already cleaned and horizontally flipped augmented directory, execute the following scripts in order:
  1. `kp_ds_gen.py` (note: change the final directory name to `kp_ds_aug_balan_3`)
  2. `kp_data_aug.py`

## **If you just want to use the datasets, download them [here](https://drive.google.com/drive/folders/1J22NMrp7-ASANnqbkdPJ8ay9WPHqV_VG?usp=sharing)**

## **If you want to use the ready-made pipeline, just run the `ipynb` notebooks in the `experiments` directory.**
