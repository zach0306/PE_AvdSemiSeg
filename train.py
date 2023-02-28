import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle
from packaging import version
import glob

from model.deeplab import Res_Deeplab
from model import deeplabv3plus
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d, FocalLoss
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet
from dataset.dataset import *
from optimizer import Ranger
import segmentation_models_pytorch as smp
from hrnet import *


import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(1)

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '400,400'
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
NUM_CLASSES = 1
NUM_STEPS = 120
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = "/home/mel/tingwei/PE_seg/seg/models/hrnet_noname/model_final_epoch_0422.pth"
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 50
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.1

PARTIAL_DATA=0.5

SEMI_START=40
LAMBDA_SEMI=0.1
MASK_T=0.1

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=0
D_REMAIN=False


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

class BCEDiceLoss(nn.Module):
    def __init__(self, type='BCEDiceLoss'):
        super().__init__()
        self.type = type
        if self.type == 'OHEMBCEDice':
            self.BCE = OhemBinaryCrossEntropy()
        elif self.type == 'OHNEMBCEDice':
            self.BCE = OhNemBinaryCrossEntropy()

    def forward(self, input, target):
        '''multi-scale inference
        ph, pw = input.size(2), input.size(3)
        h, w = target.size(1), target.size(2)
        if ph!=h or pw!=w:
            input = F.interpolate(input=input, size=(h,w), mode='bilinear')
        '''
        target = Variable(target).cuda(args.gpu)
        target = target.unsqueeze(1)
        #print(target.shape)
        if self.type == 'BCEDiceLoss':
            bce = F.binary_cross_entropy_with_logits(input, target)
        else:
            bce = self.BCE(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5*bce + dice

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)

    return D_label


def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print(input_size)

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    #model = Res_Deeplab(num_classes=args.num_classes)
    #model = deeplabv3plus.DeepLabv3_plus(nInputChannels=1, n_classes=1, os=16, pretrained=False)
    #model = smp.Unet('resnet34', classes=1, in_channels=1)
    model = hrnetv2(False, n_class=1, decoder='Original', use_softmax=False)

    # load pretrained parameters
    
    #if args.restore_from[:4] == 'http' :
    #    saved_state_dict = model_zoo.load_url(args.restore_from)
    #else:
        #saved_state_dict = torch.load(args.restore_from)
    #print('Using pre-trained......')
    #model.load_state_dict(torch.load(args.restore_from))

    # only copy the params that exist in current model (caffe-like)
    #new_params = model.state_dict().copy()
    #for name, param in new_params.items():
        #print(name)
    #    if name in saved_state_dict and param.size() == saved_state_dict[name].size():
    #        new_params[name].copy_(saved_state_dict[name])
            #print('copy {}'.format(name))
    #model.load_state_dict(new_params)
    
    model.load_state_dict(torch.load("/home/mel/tingwei/PE_seg/seg/models/hrnet_wos4_0201/model_final_epoch_0201.pth"))


    model.train()
    model.cuda(args.gpu)
    #model = nn.DataParallel(model)

    cudnn.benchmark = True

    # init D
    #model_fmap_D = fmap_FCDiscriminator(in_channal=336)
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))
    model_D.train()
    model_D.cuda(args.gpu)
    #model_fmap_D.train()
    #model_fmap_D.cuda(args.gpu)
    #model_D = nn.DataParallel(model_D)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    # load data
    folder_name = glob.glob('/home/mel/tingwei/PE_seg/open_dataset_PE/selected_t/train/*')
    random.shuffle(folder_name)

    semi_folder = glob.glob('/home/mel/tingwei/PE_seg/NCKU_dataset_PE/*')
    val_folder = glob.glob('/home/mel/tingwei/PE_seg/open_dataset_PE/selected_t/val/*')

    img_l_path=[]
    label_path=[]
    img_u_path = []
    val_img_path = []
    val_label_path = []


    for x in folder_name:
        for i in list(glob.glob(x + '/1/*dcm')):
            img_l_path.append(i)
            name = i.split('.')[0].split('/')[-1]
            label_path.append(x +'/1/'+ name +'.png')
         
    for x in semi_folder:
        for i in list(glob.glob(x + '/1/*dcm')): 
            img_u_path.append(i)

    for x in val_folder:
        for i in list(glob.glob(x + '/1/*dcm')):
            val_img_path.append(i)
            name = i.split('.')[0].split('/')[-1]
            val_label_path.append(x +'/1/'+ name +'.png')
            

    trainset_l = train_labeled(img_l_path, label_path)
    trainset_u = train_unlabeled(img_u_path)
    valset = train_labeled(val_img_path, val_label_path)
    print(len(trainset_l))
    print(len(trainset_u))

    trainloader_l = DataLoader(trainset_l,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)
    trainloader_u = DataLoader(trainset_u,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)
    valoader = DataLoader(valset,batch_size=2)



    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    #optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = Ranger(model.parameters())
    optimizer.zero_grad()

    # optimizer for discriminator network
    #optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D = Ranger(model_D.parameters())
    optimizer_D.zero_grad()

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    focal_loss = FocalLoss()
    criterion = BCEDiceLoss().cuda(args.gpu)
    #avg_meter = AverageMeter()
    #interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    #interp = F.interpolate(size=[512, 512], mode="bilinear")

    #if version.parse(torch.__version__) >= version.parse('0.4.0'):
        #interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
        #interp = F.interpolate(size=[512, 512], mode="bilinear")
    #else:
        #interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
        #interp = F.interpolate(size=[512, 512], mode="bilinear")


    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    print('Start training......')


    for i_iter in range(args.num_steps):

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0
        interp = nn.Upsample(size=(400, 400), mode='bilinear')

        avg_meters = {'loss_seg': AverageMeter(),
                      'loss_adv_pred': AverageMeter(),
                      'loss_D': AverageMeter(),
                      'loss_semi': AverageMeter(),
                      'loss_semi_adv': AverageMeter(),
                      'train_iou': AverageMeter()}

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # do semi first
            if (args.lambda_semi > 0 or args.lambda_semi_adv > 0 ) and i_iter >= args.semi_start_adv :
                
                for batch_id, (images, img_path) in enumerate(trainloader_u):

                    # only access to img
                    #images = batch
                    images = Variable(images).cuda(args.gpu)
                    #print('check1', images.shape)


                    pred, fmap = model(images)
                    #print(pred.shape)
                    #print(fmap.shape)
                    D_in = torch.cat([pred, fmap], 1)
                    pred_remain = pred.detach()

                    D_out = interp(model_D(F.softmax(D_in, dim=1)))
                    #D_out = F.interpolate(model_D(F.softmax(pred)), size=[512, 512], mode="bilinear")
                    D_out_sigmoid = F.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)

                    ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)
                    #ignore_mask_remain = torch.from_numpy(ignore_mask_remain)

                    loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
                    loss_semi_adv = loss_semi_adv/args.iter_size

                    #loss_semi_adv.backward()
                    #loss_semi_adv_value += loss_semi_adv.data.cpu().numpy()/args.lambda_semi_adv
                    avg_meters['loss_semi_adv'].update(loss_semi_adv.item(), images.size(0))
                    #avg_meters['semiadv_loss'].update(loss_semi_adv_value.item(), images.size(0)

                    if args.lambda_semi <= 0 or i_iter < args.semi_start:
                        loss_semi_adv.backward(retain_graph=True)
                        loss_semi_value = 0
                    else:
                        # produce ignore mask
                        semi_ignore_mask = (D_out_sigmoid < args.mask_T)

                        semi_gt = pred.data.cpu().numpy().argmax(axis=1)
                        semi_gt[semi_ignore_mask] = 255

                        semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
                        #print('semi ratio: {:.4f}'.format(semi_ratio))

                        if semi_ratio == 0.0:
                            loss_semi_value += 0
                        else:
                            semi_gt = torch.FloatTensor(semi_gt)

                            loss_semi = args.lambda_semi * criterion(pred, semi_gt)
                            loss_semi = loss_semi/args.iter_size
                            #loss_semi_value += loss_semi.data.cpu().numpy()/args.lambda_semi
                            avg_meters['loss_semi'].update(loss_semi.item(), images.size(0))
                            loss_semi += loss_semi_adv
                            loss_semi.backward(retain_graph=True)

                            #avg_meters['semi_loss'].update(loss_semi_value.item(), images.size(0)

            else:
                loss_semi = None
                loss_semi_adv = None

            # train with source

            for batch_id, (images,labels,_,_) in enumerate(trainloader_l):

                #images, labels, _, _ = batch
                images = Variable(images).cuda(args.gpu)
                #print('check1', images.shape)
                ignore_mask = (labels.numpy() == 255)
                #ignore_mask = torch.from_numpy(ignore_mask)
                #print('check2', ignore_mask.shape)
                pred, fmap = model(images)
                #print(pred.shape)
                #print(fmap.shape)
                D_in = torch.cat([pred, fmap], 1)
                #print(D_in.shape)
                iou = iou_score(pred, labels)
                avg_meters['train_iou'].update(iou, images.size(0))
                #print('check1', pred.shape)
                #print(labels.shape)
                #pred = F.interpolate(model(images), size=[512, 512], mode="bilinear")

                loss_seg = criterion(pred, labels)

                D_out = interp(model_D(F.softmax(D_in, dim=1)))
                #print('a', D_out.shape)
                #print('b', images.shape)
                #D_out = images*D_out
                #D_out = F.interpolate(model_D(F.softmax(pred)), size=[512, 512], mode="bilinear")

                loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

                loss = loss_seg + args.lambda_adv_pred * loss_adv_pred

                # proper normalization
                loss = loss/args.iter_size
                loss.backward(retain_graph=True)
                #loss_seg_value += loss_seg.data.cpu().numpy()/args.iter_size
                avg_meters['loss_seg'].update(loss_seg.item(), images.size(0))
                #loss_adv_pred_value += loss_adv_pred.data.cpu().numpy()/args.iter_size
                avg_meters['loss_adv_pred'].update(loss_adv_pred.item(), images.size(0))

                #avg_meters['seg_loss'].update(loss_seg_value.item(), images.size(0))
                #avg_meters['advp_loss'].update(loss_adv_pred_value.item(), images.size(0))

                # train D

                # bring back requires_grad
                for param in model_D.parameters():
                    param.requires_grad = True

                # train with pred
                pred = pred.detach()

                if args.D_remain:
                    pred = torch.cat((pred, pred_remain), 0)
                    ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0)
                    #ignore_mask = torch.from_numpy(ignore_mask)
                D_in = torch.cat([pred, fmap], 1)
                #print(pred.shape)
                D_out = interp(model_D(F.softmax(D_in, dim=1)))
                #print(D_out.shape)
                #print(images.shape)
                #D_out = images*D_out
                #D_out = F.interpolate(model_D(F.softmax(pred)), size=[512, 512], mode="bilinear")
                loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
                loss_D = loss_D/args.iter_size/2
                loss_D.backward(retain_graph=True)
                #loss_D_value += loss_D.data.cpu().numpy()
                avg_meters['loss_D'].update(loss_D.item(), images.size(0))

                #avg_meters['d_loss'].update(loss_D_value.item(), images.size(0))


            # train with gt
            # get gt labels
            for batch_id, (images,labels_gt,_,_) in enumerate(trainloader_l):

                #_, labels_gt, _, _ = batch
                images = Variable(images).cuda(args.gpu)
                pred, fmap = model(images)
                D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
                D_gt_v_f = F.interpolate(D_gt_v, (400, 400), mode='bilinear')
                ignore_mask_gt = (labels_gt.numpy() == 255)
                #ignore_mask_gt = torch.from_numpy(ignore_mask_gt)
                D_in = torch.cat([D_gt_v_f, fmap], 1)
                #print(D_in.shape)

                D_out = interp(model_D(F.softmax(D_in, dim=1)))
                #D_out = images*D_out
                #D_out = F.interpolate(model_D(D_gt_v), size=[512, 512], mode="bilinear")
                loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
                loss_D = loss_D/args.iter_size/2
                loss_D.backward(retain_graph=True)
                #loss_D_value += loss_D.data.cpu().numpy()
                avg_meters['loss_D'].update(loss_D.item(), labels_gt.size(0))

                #avg_meters['d_loss'].update(loss_D_value.item(), labels_gt.size(0))



            optimizer.step()
            optimizer_D.step()

            #print('exp = {}'.format(args.snapshot_dir))
            #print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f},  IOU = {7:.8f}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value, loss_semi_adv_value, avg_meter['train_iou'].avg))
            print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f},  IOU = {7:.8f}'.format(i_iter, args.num_steps, avg_meters['loss_seg'].avg, avg_meters['loss_adv_pred'].avg, avg_meters['loss_D'].avg, avg_meters['loss_semi'].avg, avg_meters['loss_semi_adv'].avg, avg_meters['train_iou'].avg))
            #print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f},  IOU = {7:.3f}'.format(i_iter, args.num_steps, avg_meter['seg_loss'].avg, avg_meter['advp_loss'].avg, avg_meter['d_loss'].avg, avg_meter['semi_loss'].avg, avg_meter['semiadv_loss'].avg, avg_meter['iou'].avg))


            if i_iter >= args.num_steps-1:
                print('save model ...')
                torch.save(model.state_dict(),osp.join(args.snapshot_dir, '0430g_hr_hu_wos4_cmuh'+str(args.num_steps)+'.pth'))
                torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, '0430d_fc_hr_hu_wos4_cmuh'+str(args.num_steps)+'.pth'))
                break

            if i_iter % args.save_pred_every == 0 and i_iter!=0:
                print('taking snapshot ...')
                torch.save(model.state_dict(),osp.join(args.snapshot_dir, '0430g_hr_hu_wos4_cmuh_every'+str(i_iter)+'.pth'))
                torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, '0430d_fc_hr_hu_wos4_cmuh_every'+str(i_iter)+'.pth'))

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
