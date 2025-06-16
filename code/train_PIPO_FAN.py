"""
PIPO-FAN training
"""
import argparse
import logging
import os
import tqdm
import random
import time
import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch import nn
from torch.nn import DataParallel
import SimpleITK as sitk

from dataloader.WORD_dataloader3d import Word3D
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.PIPO_FAN.concave_dps_w import ResUNet
from models.PIPO_FAN.utils import *
from models.weight_init import initialize_weights
from dataloader import WORD_transforms as tr
from utils.average_meter import AverageMeter
from utils.evaluation_seg import *
from utils.losses import *
from utils.val3D import test_single_case
from tensorboardX import SummaryWriter

# -------------------- reproduction ------------------------ #
# torch.cuda.current_device()  # no problem in server
# torch.cuda._initialized = True  # no problem in server
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

CLASS_NAME = ['Liver', 'Spleen', 'Kidney(L)', 'Kidney(R)', 'Stomach', 'Gallbladder', 'Esophagus',
               'Pancreas', 'Duodenum', 'Colon', 'Intestine', 'Adrenal', 'Rectum', 'Bladder', 'Head of Femur(L)', 'Head of Femur(R)']

class CrossEntropyLoss3d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss3d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)


def parse_args():
    desc = "Pytorch implementation of PIPO_FAN"
    parser = argparse.ArgumentParser(description=desc)
    # dir config
    parser.add_argument('--exp_dir', type=str, default='./exp/PIPO_FAN')
    parser.add_argument('--data_dir', type=str, default='./dataset/word3d')
    parser.add_argument('--workspace', type=str, default='./exp/PIPO_FAN/checkpoint')
    # GPU config
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--gpu_grop', type=int, default=[0, 1])
    # training config
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=list, default=[128, 128, 96])
    parser.add_argument('--num_classes', type=int, default=17)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoches', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('-wi', '--weight_init', type=str, default="xavier",
                        help='Weight initialization method, or path to weights file '
                             '(for fine-tuning or continuing training)')
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=25)

    return parser.parse_args()


def validate_slice(model, dataloader, args, writer, epoch):
    training = model.training
    model.eval()
    val_dice = AverageMeter()
    with torch.no_grad():
        for sample in tqdm.tqdm(dataloader, total=len(dataloader), ncols=80, leave=False):

            image = Variable(sample['image'].squeeze(dim=0).squeeze(dim=0).cuda())
            target = sample['label'].cuda()
            
            target = Variable(target)

            pred_seg = test_single_case(model, image, stride_xy=args.patch_size[0], stride_z=args.patch_size[2],
                                        patch_size=args.patch_size, num_classes=args.num_classes)

            gt_volumn = target.squeeze(dim=0)
            dice_score = get_multi_class_evaluation_score(pred_seg, gt_volumn.cpu().numpy(),
                                                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], False, 'dice')
            val_dice.update(torch.tensor(dice_score))

            if (epoch + 1) % (args.print_interval + 19) == 0:
                image_v = image.unsqueeze(0).permute(0, 2, 3, 1)
                image_v = image_v[0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image_v, 5, normalize=True)
                writer.add_image('train/Image', grid_image, epoch)
                
                seg_pred_v = torch.from_numpy(pred_seg).permute(1, 2, 0)
                seg_pred_v = seg_pred_v[:, :, 20:61:10].permute(2, 0, 1)
                pre_v = seg_pred_v.unsqueeze(dim=1).repeat(1, 3, 1, 1)
                grid_image = make_grid(pre_v, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, epoch)

                label_v = gt_volumn.permute(1, 2, 0)
                label_v = label_v[:, :, 20:61:10].permute(2, 0, 1)
                label_v = label_v.unsqueeze(dim=1).repeat(1, 3, 1, 1)
                grid_image = make_grid(label_v, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, epoch)
    if training:
        model.train()

    return val_dice


def train(model, train_loader, val_loader, writer, args):
    # define the criterion
    criterion = CrossEntropyLoss3d().cuda()

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=4, verbose=True,
                                                     min_lr=1e-4)
    
    best_dice = torch.zeros(17)
    best_epoch = 1

    for epoch in range(args.start_epoch, args.epoches):
        seg_loss_epoch = AverageMeter()

        # train in each epoch
        for batch_idx, sample in tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc='Train epoch=%d' % epoch, ncols=80, leave=False):
            model.train()

            image = sample['image'].cuda()
            label = sample['onehot_label'].cuda()
            cur_task = sample['cur_task']
            cur_task = [[row[i] for row in cur_task] for i in range(len(cur_task[0]))]

            seg_pred = model(image)
            seg_pred = torch.softmax(seg_pred, dim=1)
            seg_pred = torch.clamp(seg_pred, min=1e-10, max=1)

            optimizer.zero_grad()
            
            # TALoss
            target_onehot = partial_onehot_sample(label, cur_task, CLASS_NAME)
            merged_pre = merge_prediction(
                        seg_pred, target_onehot, cur_task, CLASS_NAME)
            target = torch.argmax(target_onehot, dim=1)
            seg_loss = criterion(merged_pre, target)

            seg_loss_epoch.update(seg_loss.cpu())

            # backward the gradient
            seg_loss.backward()
            optimizer.step()

        # print training result
        logging.info(
            '\n Epoch[%4d/%4d]-Lr: %.6f -->for ct_array in ct_array_list: Train...' % (epoch + 1, args.epoches, optimizer.param_groups[0]['lr']))
        logging.info(
                '\t Seg Loss = %.4f' % (seg_loss_epoch.avg))

        # tensorboard
        if (epoch + 1) % args.print_interval == 0:
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalars('Train/Losses',{'seg': seg_loss_epoch.avg}, epoch)

        # validate and visualization
        if not os.path.exists(args.workspace):
            os.mkdir(args.workspace)
        result_dir = os.path.join(args.workspace, 'val_results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        model_dir = os.path.join(args.workspace, 'models')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if (epoch + 1) % args.val_interval == 0:
            val_dice = validate_slice(model, val_loader, args, writer, epoch)
            
            logging.info('\n Epoch[%4d/%4d] --> Valid...' % (epoch + 1, args.epoches))
            logging.info(
                '\t [Dice Coef: mean=%.4f, BG=%.4f, Liver=%.4f, Spleen=%.4f, LK=%.4f, RK=%.4f, Stomach=%.4f, Gallb=%.4f, Esopha=%.4f, Pancreas=%.4f, Duode=%.4f, Colon=%.4f, Intes=%.4f, Adrenal=%.4f, Rectum=%.4f, Bladder=%.4f, LH=%.4f, RH=%.4f]' %
                (torch.mean(val_dice.avg), val_dice.avg[0], val_dice.avg[1], val_dice.avg[2], val_dice.avg[3], val_dice.avg[4], val_dice.avg[5], val_dice.avg[6],
                 val_dice.avg[7], val_dice.avg[8], val_dice.avg[9], val_dice.avg[10], val_dice.avg[11], val_dice.avg[12], val_dice.avg[13], val_dice.avg[14], val_dice.avg[15], val_dice.avg[16]))
            writer.add_scalars('Val/Dice',
                                {'Liver': val_dice.avg[1], 'Spleen': val_dice.avg[2],
                                'LK': val_dice.avg[3], 'RK': val_dice.avg[4], 'Stomach' :val_dice.avg[5],
                                'Gallb': val_dice.avg[6], 'Esopha': val_dice.avg[7], 'Pancreas': val_dice.avg[8],
                                'Duode': val_dice.avg[9], 'Colon': val_dice.avg[10], 'Intes': val_dice.avg[11],
                                'Adrenal': val_dice.avg[12], 'Rectum': val_dice.avg[13], 'Bladder': val_dice.avg[14],
                                'LH': val_dice.avg[15], 'ROH': val_dice.avg[16],'BG': val_dice.avg[0],
                                'mean': torch.mean(val_dice.avg)}, epoch)
            # save best model
            if torch.mean(val_dice.avg) >= torch.mean(best_dice):
                best_model_path = os.path.join(model_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(
                    '\n [Epoch[%4d/%4d] --> Dice improved from %.4f in epoch %4d to %.4f]' %
                    (epoch + 1, args.epoches, torch.mean(best_dice), best_epoch, torch.mean(val_dice.avg)))
                best_dice, best_epoch = val_dice.avg, epoch + 1
            else:
                logging.info('\n [Epoch[%4d/%4d] --> Dice did not improved with %.4f in epoch %d)]' %
                                (epoch + 1, args.epoches, torch.mean(best_dice), best_epoch))
            
            # check for plateau
            dice_sum = 0
            dice_sum += torch.mean(val_dice.avg)
            scheduler.step(dice_sum)

    # save final model
    final_model_path = os.path.join(model_dir, 'final_model.pth')
    torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()}, final_model_path)
    logging.info('\t [Save Final Model] to %s' % (final_model_path))


def main():
    args = parse_args()

    # GPU Setting
    # single GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # # GPU Parallel
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

    # define logger
    os.makedirs(args.exp_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.exp_dir, 'train.log'), level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # print all parameters
    for name, v in vars(args).items():
        logging.info(name + ': ' + str(v))

    # dataset
    composed_transforms_tr = transforms.Compose([
        tr.RemainClass(CLASS_NAME),
        tr.WordTrainerCrop(args.patch_size),
        tr.CreateOnehotLabel(args.num_classes),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.ToTensor()
    ])

    # dataloader config
    train_set = Word3D(nii_dir=args.data_dir, mode='train', transform=composed_transforms_tr)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=True)
    valid_set = Word3D(nii_dir=args.data_dir, mode='val', transform=composed_transforms_ts)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    #  init model
    model = ResUNet(args.in_channel, args.num_classes).cuda()

    print('parameter numer:', sum([p.numel() for p in model.parameters()]))

    # GPU Parallel
    # if torch.cuda.device_count() > 1:
    #     model = DataParallel(model, device_ids=args.gpu_grop)

    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print('Resume finished!')
    else:
        initialize_weights(model, args.weight_init)

    # summary writer config
    writer = SummaryWriter(log_dir=args.exp_dir, comment=args.exp_dir.split('/')[-1])

    # train
    train(model, train_loader, valid_loader, writer, args)


if __name__ == '__main__':
    main()

