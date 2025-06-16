'''
PIPO-FAN testing
'''
import os
import pytz
import tqdm
import torch
import random
import argparse
import numpy as np
import SimpleITK as sitk

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader.WORD_dataloader3d import Word3D
from dataloader import WORD_transforms as tr
from utils.losses import *
from utils.evaluation_seg import *
from datetime import datetime
from models.PIPO_FAN.concave_dps_w import ResUNet
from utils.val3D import test_single_case

# -------------------- reproduction ------------------------ #
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)  # Numpy module.
random.seed(42)  # Python random module.
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.set_default_tensor_type('torch.FloatTensor')

CLASS_NAME = ['Background', 'Liver', 'Spleen', 'Kidney(L)', 'Kidney(R)', 'Stomach', 'Gallbladder', 'Esophagus',
               'Pancreas', 'Duodenum', 'Colon', 'Intestine', 'Adrenal', 'Rectum', 'Bladder', 'Head of Femur(L)', 'Head of Femur(R)']

def main():
    parser = argparse.ArgumentParser()
    # dir config
    parser.add_argument('--exp_dir', type=str, default='./exp/PIPO_FAN')
    parser.add_argument('--data_dir', type=str, default='./dataset/word3d')
    # GPU file
    parser.add_argument('-g', '--gpu', type=int, default=0)
    # test config
    parser.add_argument('--model_file', type=str, default='checkpoint/models/best_model.pth', help='Model path')
    parser.add_argument('--dataset', type=str, default='test', help='test folder id contain images ROIs to test')
    parser.add_argument('--patch_size', type=list, default=[128, 128, 96])
    parser.add_argument('--save_imgs', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=17)
    parser.add_argument('--in_channel', type=int, default=1)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = os.path.join(args.exp_dir, args.model_file)
    output_path = os.path.join(args.exp_dir, 'test')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1. dataset
    composed_transforms_ts = transforms.Compose([
        tr.ToTensor()
    ])

    db_test = Word3D(nii_dir=args.data_dir, mode='test', transform=composed_transforms_ts)
    test_loader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 2. model
    model = ResUNet(args.in_channel, args.num_classes).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    
    print('==> Loading model file: %s' % (model_file))
    # model_data = torch.load(model_file)

    checkpoint = torch.load(model_file)
    pretrained_dict = checkpoint                 # for best model
    # pretrained_dict = checkpoint['model']         # for final checkpoint model
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    val_class_dice = [[] for _ in range(17)]
    total_asd_class = [[] for _ in range(17)]
    total_num = 0
    timestamp_start = datetime.now(pytz.timezone('Asia/Hong_Kong'))

    model.eval()
    for batch_idx, sample in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),ncols=80, leave=False):
        
        image = Variable(sample['image'].squeeze(dim=0).squeeze(dim=0).cuda())
        target = Variable(sample['label'].cuda())
        img_name = sample['img_name'][0]

        image_sam = sitk.ReadImage(os.path.join(args.data_dir, 'imagesTs', img_name))
        data_spacing = image_sam.GetSpacing()

        pred_seg = test_single_case(model, image, stride_xy=64, stride_z=64,
                                    patch_size=args.patch_size, num_classes=args.num_classes)

        gt_volumn = target.squeeze(dim=0)
        dice_score = get_multi_class_evaluation_score(pred_seg, gt_volumn.cpu().numpy(),
                                                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], False, 'dice')
        assd_score = get_multi_class_evaluation_score(pred_seg, gt_volumn.cpu().numpy(),
                                                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], False, 'assd', data_spacing)
        
        # Dice Score
        for i in range(len(val_class_dice)):
            val_class_dice[i].append(dice_score[i])

        # ASSD Score
        for i in range(len(total_asd_class)):
            total_asd_class[i].append(assd_score[i])

        total_num += 1

        if args.save_imgs:
            pred_img = sitk.GetImageFromArray(pred_seg.astype('int32'))
            pred_img.SetOrigin(image_sam.GetOrigin())
            pred_img.SetSpacing(image_sam.GetSpacing())
            pred_img.SetDirection(image_sam.GetDirection())
            path0 = os.path.join(output_path, 'test_results', img_name.split('.')[0] +'_pred.nii.gz')
            if not os.path.exists(os.path.dirname(path0)):
                os.makedirs(os.path.dirname(path0))
            sitk.WriteImage(pred_img, path0)

    # log dice score
    for i in range(len(val_class_dice)):
        print('==>WORD %s Dice Score: %s\n' % (CLASS_NAME[i], val_class_dice[i]))


    import csv
    with open(output_path+'/Dice_results.csv', 'a+') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(['Result in: ' + args.model_file])
        for i in range(len(val_class_dice)):
            wr.writerow([CLASS_NAME[i]])
            for index in range(len(val_class_dice[i])):
                wr.writerow([torch.from_numpy(np.array([val_class_dice[i][index]]))])        


    val_class_dice_mean = []
    val_class_dice_std = []
    asd_class_mean = []
    asd_class_std = []
    for i in range(len(val_class_dice)):
        val_class_dice_mean.append(np.mean(val_class_dice[i]))
        val_class_dice_std.append(np.std(val_class_dice[i]))
        asd_class_mean.append(np.mean(total_asd_class[i]))
        asd_class_std.append(np.std(total_asd_class[i]))

    total_dice = []
    for i in range(1, len(val_class_dice)):
        total_dice += val_class_dice[i]
    total_dice_mean = np.mean(total_dice)
    total_dice_std = np.std(total_dice)  
    
    total_asd = []
    for i in range(1, len(total_asd_class)):
        total_asd += total_asd_class[i]
    total_asd_mean = np.mean(total_asd)
    total_asd_std = np.std(total_asd)

    for i in range(len(val_class_dice_mean)):
        print('''\n==>val_{0}_dice : {1}-{2}'''.format(CLASS_NAME[i], val_class_dice_mean[i], val_class_dice_std[i]))
    print('''\n==>val_Average_dice : {0}-{1}'''.format(total_dice_mean, total_dice_std))

    for i in range(len(asd_class_mean)):
        print('''\n==>ave_asd_{0} : {1}-{2}'''.format(CLASS_NAME[i], asd_class_mean[i], asd_class_std[i]))
    print('''\n==>val_Average_asd : {0}-{1}'''.format(total_asd_mean, total_asd_std))


if __name__ == '__main__':
    main()
