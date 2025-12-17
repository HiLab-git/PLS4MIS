# 📊 Datasets & Challenges for Partially Labeled Medical Image Segmentation

This repository provides a curated list of **publicly available datasets** for *partially labeled medical image segmentation*.  
In particular, **pre-processed versions of the WORD dataset** are released to support immediate experimental use. The pre-processed version of the FLARE2023 dataset is still awaiting open-source approval from the organizers; stay tuned.

---

## 1️⃣ WORD Dataset

📁 **Dataset Information:** [WORD](WORD)

### Overview
- The WORD dataset contains **150 abdominal CT volumes**, split into:
  - **100** for training  
  - **20** for validation  
  - **30** for testing
- Each CT volume consists of **159–330 slices** with a spatial resolution of **512 × 512 pixels**.
- Imaging properties:
  - In-plane resolution: **0.976 mm × 0.976 mm**
  - Inter-slice spacing: **2.5–3.0 mm**
- To simulate partial labeling, we define  
  \[
  r \in \{2/16, 4/16, 6/16\}
  \]
  where each training volume contains annotations for **16 × r randomly selected organs**.
- Additional details can be found at the official [WORD dataset page](WORD).

---

### 📥 Dataset Download

The processed **partially labeled WORD dataset** (with labeling ratios of 2/16, 4/16, and 6/16) is publicly available via the following options:

1. **Baidu Disk (China)**  
   - Link: https://pan.baidu.com/s/1wRiQGlArH6KKq2ZB03BMfw  
   - Password: **WDPL**

2. **Google Drive**  
   - Link: https://drive.google.com/file/d/1v29xcg7SpQUTRNFcU17PXguMDa9wG9Fo/view?usp=drive_link  

---


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
- Folders with the **'_LeafDice'** suffix (WORD_LeafDice) indicate data preprocessed specifically for the LeafDice method and should **only be used when running LeafDice-related experiments**.
- 'labelsTr_2', 'labelsTr_4', and 'labelsTr_6' correspond to labeling ratios of 2/16, 4/16, and 6/16, respectively. When conducting experiments with these ratios, **please rename the corresponding folder to 'labelsTr'** so that it can be correctly read by the Dataset function.

---

## [2. FLARE2023 Challenge]()
