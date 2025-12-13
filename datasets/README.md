# 📊 Datasets/Challenges for Partially Labeled Medical Image Segmentation
A list of publicly available, partially labeled medical image segmentation datasets is included. Pre-processed WORD dataset versions are provided for immediate use in experiments.

## [WORD Dataset](WORD)
* This dataset consists of 150 abdominal Computed Tomography (CT) volumes, divided into 100 volumes for training, 20 for validation, and 30 for testing.
* Each CT volume contains 159–330 slices of 512 × 512 pixels, with an in-plane resolution of 0.976 mm × 0.976 mm and inter-slice spacing of 2.5-3.0 mm.
* We used r ∈ {2∕16, 4∕16, 6∕16} to denote the ratio of labeled classes in a volume and randomly select labels for 16 × r organs in each training volume.
* More details of this dataset can be found at [this](WORD)

### Dataset folder structure
Now, the processed partially labeled WORD dataset, with class ratios of 2∕16, 4∕16, and 6∕16, is publicly available. It can be downloaded through the following two options:
1) Download the data from [BaiduDisk](https://pan.baidu.com/s/1d0cFhj3LU029oHajNni8KQ), the password is *WORDPL*.
2) Using your Google account to download the data ([Goole Driven](https://drive.google.com/drive/folders/1i2xbXxdEYnjNZVUtGZxYdwaeKmNmywnY)).
