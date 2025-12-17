##########
# Ours ###
##########

# Train WORD (Annotation scale: 2/16)
# python -u train_unet.py --exp_dir ./exp/WORD/UNet_P2 --data_dir ./datasets/WORD \
#        --workspace ./exp/WORD/UNet_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
#        --num_classes 17 --epoches 500 --learning_rate 0.01
# python -u train_MEloss.py --exp_dir ./exp/WORD/MEL_P2 --data_dir ./datasets/WORD \
#        --workspace ./exp/WORD/MEL_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
#        --num_classes 17 --epoches 500 --learning_rate 0.01
# python -u train_LeafDice.py --exp_dir ./exp/WORD/LeafDice_P2 --data_dir ./datasets/WORD_LeafDice \
#        --workspace ./exp/WORD/LeafDice_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
#        --num_classes 17 --epoches 500 --learning_rate 0.01
# python -u train_PLSeg.py --exp_dir ./exp/WORD/PLSeg_P2 --data_dir ./datasets/WORD \
#        --workspace ./exp/WORD/PLSeg_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
#        --num_classes 17 --epoches 500 --learning_rate 0.01
# python -u train_PIPO_FAN.py --exp_dir ./exp/WORD/PIPO_FAN_P2 --data_dir ./datasets/WORD \
#        --workspace ./exp/WORD/PIPO_FAN_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
#        --num_classes 17 --epoches 500 --learning_rate 0.01
# python -u train_DoDNet.py --exp_dir ./exp/WORD/DoDNet_P2 --data_dir ./datasets/WORD \
#        --workspace ./exp/WORD/DoDNet_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
#        --num_classes 17 --epoches 1000 --learning_rate 0.01
# python -u train_DoDNet_WORD.py --exp_dir ./exp/WORD/DoDNet_P2 --data_dir ./datasets/WORD \
#        --workspace ./exp/WORD/DoDNet_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
#        --num_classes 17 --epoches 500 --learning_rate 0.01
# python -u train_CoNeMOS.py --exp_dir ./exp/WORD/CoNeMOS_P2 --data_dir ./datasets/WORD \
#        --workspace ./exp/WORD/CoNeMOS_P2/checkpoint --gpu 0 --batch_size 4 --patch_size 128 128 96 \
#        --num_classes 17 --epoches 1000 --learning_rate 0.01


# Train FLARE2023
# python -u train_unet.py --exp_dir ./exp/FLARE2023/UNet --data_dir ./datasets/FLARE2023 \
#        --workspace ./exp/FLARE2023/UNet/checkpoint --gpu 1 --batch_size 4 --patch_size 128 128 64 \
#        --num_classes 13 --epoches 600 --learning_rate 0.01
# python -u train_MEloss.py --exp_dir ./exp/FLARE2023/MEL --data_dir ./datasets/FLARE2023 \
#        --workspace ./exp/FLARE2023/MEL/checkpoint --gpu 1 --batch_size 4 --patch_size 128 128 64 \
#        --num_classes 13 --epoches 600 --learning_rate 0.01
# python -u train_LeafDice.py --exp_dir ./exp/FLARE2023/LeafDice --data_dir ./datasets/FLARE2023_LeafDice \
#        --workspace ./exp/FLARE2023/LeafDice/checkpoint --gpu 1 --batch_size 4 --patch_size 128 128 64 \
#        --num_classes 13 --epoches 600 --learning_rate 0.01
# python -u train_PLSeg.py --exp_dir ./exp/FLARE2023/PLSeg --data_dir ./datasets/FLARE2023 \
#        --workspace ./exp/FLARE2023/PLSeg/checkpoint --gpu 1 --batch_size 4 --patch_size 128 128 64 \
#        --num_classes 13 --epoches 600 --learning_rate 0.01
# python -u train_PIPO_FAN.py --exp_dir ./exp/FLARE2023/PIPO_FAN --data_dir ./datasets/FLARE2023 \
#        --workspace ./exp/FLARE2023/PIPO_FAN/checkpoint --gpu 1 --batch_size 4 --patch_size 128 128 64 \
#        --num_classes 13 --epoches 600 --learning_rate 0.01
# python -u train_DoDNet.py --exp_dir ./exp/FLARE2023/DoDNet --data_dir ./datasets/FLARE2023 \
#        --workspace ./exp/FLARE2023/DoDNet/checkpoint --gpu 1 --batch_size 4 --patch_size 128 128 64 \
#        --num_classes 13 --epoches 1200 --learning_rate 0.01
# python -u train_CoNeMOS.py --exp_dir ./exp/FLARE2023/CoNeMOS --data_dir ./datasets/FLARE2023 \
#        --workspace ./exp/FLARE2023/CoNeMOS/checkpoint --gpu 1 --batch_size 4 --patch_size 128 128 64 \
#        --num_classes 13 --epoches 1200 --learning_rate 0.01



# Test WORD
# python -u test_PLSeg.py --exp_dir ./exp/WORD/PLSeg_P2 --data_dir ./datasets/WORD \
#        --gpu 1 --patch_size 128 128 96 --batch_size 1 --num_classes 17 --stride_xy 64 --stride_z 64


# Test FLARE2023
# python -u test_PLSeg.py --exp_dir ./exp/FLARE2023/PLSeg --data_dir ./datasets/FLARE2023 \
#        --gpu 1 --patch_size 128 128 64 --batch_size 1 --num_classes 13 --stride_xy 32 --stride_z 32