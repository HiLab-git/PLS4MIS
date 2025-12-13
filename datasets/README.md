# 📊 Datasets/Challenges for Partially Labeled Medical Image Segmentation
A list of publicly available, partially labeled medical image segmentation datasets is included. Pre-processed WORD dataset versions are provided for immediate use in experiments.

---

## [1. WORD Dataset](WORD)
* This dataset consists of 150 abdominal Computed Tomography (CT) volumes, divided into 100 volumes for training, 20 for validation, and 30 for testing.
* Each CT volume contains 159–330 slices of 512 × 512 pixels, with an in-plane resolution of 0.976 mm × 0.976 mm and inter-slice spacing of 2.5-3.0 mm.
* We used r ∈ {2∕16, 4∕16, 6∕16} to denote the ratio of labeled classes in a volume and randomly select labels for 16 × r organs in each training volume.
* More details of this dataset can be found at [this](WORD)

### Dataset folder structure
Now, the processed partially labeled WORD dataset, with class ratios of 2∕16, 4∕16, and 6∕16, is publicly available. It can be downloaded through the following two options:
1) Download the data from [BaiduDisk (China)](https://pan.baidu.com/s/1wRiQGlArH6KKq2ZB03BMfw), the password is *WDPL*.
2) Using your Google account to download the data ([Google Driven](https://drive.google.com/file/d/1v29xcg7SpQUTRNFcU17PXguMDa9wG9Fo/view?usp=drive_link)).

Datasets must be located in the `/code/datasets` folder. Each segmentation dataset is organized into two separate folders. For example, the WORD dataset includes two folders: *WORD* and *WORD_LeafDice*.
WORD dataset is stored in the `/code/datasets` folder like this:

    /code/datasets/
    ├── WORD/
    │   ├── imagesTr
    │   ├── imagesTs
    │   ├── imagesVal
    │   ├── labelsTr_2
    │   ├── labelsTr_4
    │   ├── labelsTr_6
    │   ├── labelsTr_All
    │   ├── labelsTs
    │   ├── labelsVal
    ├── WORD_LeafDice/
    │   ├── imagesTr
    │   ├── imagesTs
    │   ├── imagesVal
    │   ├── labelsTr_2
    │   ├── labelsTr_4
    │   ├── labelsTr_6
    │   ├── labelsTr_All
    │   ├── labelsTs
    │   ├── labelsVal
    ├── ...

**Remember ( ! ! ! ):** 
- Folders with the **'_LeafDice'** suffix indicate data preprocessed specifically for the LeafDice method and should **only be used when running LeafDice-related experiments**.
- 'labelsTr_2', 'labelsTr_4', and 'labelsTr_6' correspond to labeling ratios of 2/16, 4/16, and 6/16, respectively. When conducting experiments with these ratios, **please rename the corresponding folder to 'labelsTr'** so that it can be correctly read by the Dataset function.

---

## [2. FLARE2023 Dataset]()
