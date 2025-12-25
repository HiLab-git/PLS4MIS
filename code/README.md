# 🔬 Partially Labeled Supervision for Medical Image Segmentation (**PLS4MIS**)

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=1.13.0
* TensorBoardX
* Python == 3.9
* Some basic Python packages such as Numpy, Scikit-image, SimpleITK, Scipy, Matplotlib, Medpy......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

## Usage

1. Clone the repo:
```
git clone https://github.com/HiLab-git/PLS4MIS.git
cd PLS4MIS/
```
2. Download the processed dataset and put the data in `/code/datasets` folder. For more detailed data processing information, please refer to [this](../datasets).

3. Train the model
```
cd code/

# Train WORD (Annotation scale: 2/16)
python -u train_PLSeg.py --exp_dir ./exp/WORD/PLSeg_P2 --data_dir ./datasets/WORD \
       --workspace ./exp/WORD/PLSeg_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
       --num_classes 17 --epoches 500 --learning_rate 0.01

# Train FLARE2023
python -u train_PLSeg.py --exp_dir ./exp/FLARE2023/PLSeg --data_dir ./datasets/FLARE2023 \
       --workspace ./exp/FLARE2023/PLSeg/checkpoint --gpu 1 --batch_size 4 --patch_size 128 128 64 \
       --num_classes 13 --epoches 600 --learning_rate 0.01
```
4. Test the model
```
# Test WORD
python -u test_PLSeg.py --exp_dir ./exp/WORD/PLSeg_P2 --data_dir ./datasets/WORD \
       --gpu 1 --patch_size 128 128 96 --batch_size 1 --num_classes 17 --stride_xy 64 --stride_z 64


# Test FLARE2023
python -u test_PLSeg.py --exp_dir ./exp/FLARE2023/PLSeg --data_dir ./datasets/FLARE2023 \
       --gpu 1 --patch_size 128 128 64 --batch_size 1 --num_classes 13 --stride_xy 32 --stride_z 32
```
5. For more training information on the other model, please refer to `run.sh`. The original code corresponds to `train_XXXXX.py and test_XXXXX.py`.

## Reimplemented methods

## Acknowledgement
* Part of the code is adapted from open-source codebases and original implementations of algorithms; we thank these author for their fantastic and efficient codebase, such as, [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [UA-MT](https://github.com/yulequan/UA-MT). 
