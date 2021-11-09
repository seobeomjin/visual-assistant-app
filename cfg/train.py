"""
1. supervised loss 
    label이 있는 data에 대해 supervised loss를 학습합니다. 
    
2. consistency loss 
    동일한 이미지에 대해 augmentation을 가하더라도, 예측에 대한 값은 동일해야 한다는 
    regularization을 통해 consistency loss 를 학습합니다.

3. self supervised loss
    3-1. Barlow Twins method 
    3-2. co-attention based sample relation signal
"""

"""
- EMA 안씀 
- Barlow Twins term 으로 loss term 수정 
- co-attention based SR loss term 수정

"""

import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import StepLR


from networks.models import DenseNet121
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
# from dataloaders import  dataset
# from dataloaders.dataset import TwoStreamBatchSampler
import dataset 
from dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics

from PIL import Image, ImageOps, ImageFilter

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'): 
            return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../../COVID-CT/Images-processed/', help='dataset root dir')
parser.add_argument('--txt_file_train', type=str, default='../../COVID-CT/Data-split/COVID/trainCT_COVID.txt', help='training set csv file')
# /content/drive/MyDrive/Colab Notebooks/SWC/COVID-CT/Data-split/COVID/trainCT_COVID.txt
parser.add_argument('--txt_file_val', type=str, default='./data/skin/validation.csv', help='validation set csv file')
parser.add_argument('--txt_file_test', type=str, default='./data/skin/testing.csv', help='testing set csv file')
parser.add_argument('--exp', type=str,  default='Covid_CoSR_BT', help='model_name')
parser.add_argument('--epochs', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--labeled_num', type=int, default=85, help='number of labeled')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### costs
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--consistency_relation_weight', type=int,  default=1, help='consistency relation weight')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')

# # # # # # # # # # # # # 
# for Barlow Twins signal
parser.add_argument('--bt_loss_weight', type=int,  default=1, help='Barlow Twins weight')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='4096-4096-4096', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--scale-loss', default=1 / 256, type=float,
                    metavar='S', help='scale the loss')
parser.add_argument('--apply_all_loss', default=20, type=int, 
                    help='which epoch will you apply all losses during train')
parser.add_argument('--bt_first', default=False, type=str2bool,
                    help='whether bt_loss + cls loss or not')
parser.add_argument('--bt_loss', default=False, type=str2bool,
                    help='whether bt_loss or not')
parser.add_argument('--aug_more', default=False, type=str2bool,
                    help='if False, use original augmentations')                    

args = parser.parse_args()

snapshot_path = "../model/" + args.exp + "/"

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
labeled_bs = args.labeled_bs * len(args.gpu.split(','))

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# here is the 
    ###  Mean Teacher framework base + SR loss added ### 
    
# What I have to change 
    # 1. model 
    #     - BUT now BT forwards "loss value"
    #     - 여기에다가 더해서 SR return ? 
    # 2. not use EMA ! 

if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # use ema
    def create_model(ema=False):
        # Network definition
        net = DenseNet121(out_size=dataset.N_CLASSES, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        # net = CoSR_BT(args)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, 
    #                              betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = StepLR(optimizer, step_size=1)

    
    print("models are created")

    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        base_lr = optimizer.param_groups[0]['lr']
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # dataset
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                                     
    
    print("Data set loading...")
    if args.aug_more : 
        print("advanced augmentation settings would be used")
        
        train_dataset = dataset.CovidCTDataset(root_dir=args.root_path,
                                                txt_COVID='../../../COVID-CT/Data-split/COVID/trainCT_COVID.txt',
                                                txt_NonCOVID= '../../../COVID-CT/Data-split/NonCOVID/trainCT_NonCOVID.txt',
                                                transform=dataset.TransformTwice(transforms.Compose([
                                                    transforms.Resize((224, 224)),
                                                    #transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomApply(
                                                        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                                                saturation=0.2, hue=0.1)],
                                                        p=0.2
                                                    ),
                                                    transforms.RandomGrayscale(p=0.1),
                                                    GaussianBlur(p=0.2),
                                                    Solarization(p=0.0),
                                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                    # transforms.RandomRotation(10),
                                                    # transforms.RandomResizedCrop(224),
                                                    transforms.ToTensor(),
                                                    normalize,
                                                ])))
    else : 
        print("origianl augmentation settings would be used")
        
        train_dataset = dataset.CovidCTDataset(root_dir=args.root_path,
                                                txt_COVID='../../../COVID-CT/Data-split/COVID/trainCT_COVID.txt',
                                                txt_NonCOVID= '../../../COVID-CT/Data-split/NonCOVID/trainCT_NonCOVID.txt',
                                                transform=dataset.TransformTwice(transforms.Compose([
                                                # transforms.Resize(256),
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                            #     transforms.RandomRotation(90),
                                                # random brightness and random contrast
                                                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                                transforms.ToTensor(),
                                                normalize
                                                ])))
                                            
                                            
    val_dataset = dataset.CovidCTDataset(root_dir=args.root_path,
                                          txt_COVID='../../../COVID-CT/Data-split/COVID/valCT_COVID.txt',
                                          txt_NonCOVID= '../../../COVID-CT/Data-split/NonCOVID/valCT_NonCOVID.txt',
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.CovidCTDataset(root_dir=args.root_path,
                                          txt_COVID='../../../COVID-CT/Data-split/COVID/testCT_COVID.txt',
                                          txt_NonCOVID= '../../../COVID-CT/Data-split/NonCOVID/testCT_NonCOVID.txt',
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 425)) # total training num 425
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                  num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)#, worker_init_fn=worker_init_fn)
                                
    print("Data is loaded via dataloader")
    
    model.train()

    loss_fn = losses.cross_entropy_loss()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = args.global_step
    lr_ = base_lr
    model.train()

    #train
    print("Training...")
    if args.bt_first : 
        print(f"During first {args.apply_all_loss} epochs, the model will be updated with only classification loss + BT loss")
    else : 
        print(f"During first {args.apply_all_loss} epochs, the model will be updated with only classification loss")
    print("and after then, the model will be updated with rest of loss terms ")
    for epoch in range(args.start_epoch, args.epochs):
        meters_loss = MetricLogger(delimiter="  ")
        meters_loss_classification = MetricLogger(delimiter="  ")
        meters_loss_consistency = MetricLogger(delimiter="  ")
        meters_loss_consistency_relation = MetricLogger(delimiter="  ")
        meters_loss_bt = MetricLogger(delimiter="  ")
        
        time1 = time.time()
        iter_max = len(train_dataloader)    
    
        for i, batch in enumerate(train_dataloader):
            image_batch = batch['img'][0]
            ema_image_batch = batch['img'][1]
            label_batch = batch['label']
            
            image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
            ema_inputs = ema_image_batch #+ noise2
            inputs = image_batch #+ noise1

            activations, outputs = model(inputs)
            
            with torch.no_grad():
                ema_activations, ema_outputs = ema_model(ema_inputs)

            ## MT loss (have no effect in the beginneing)
            if args.ema_consistency == 1:
                consistency_weight = get_current_consistency_weight(epoch)
                consistency_dist = torch.sum(losses.softmax_mse_loss(outputs, ema_outputs)) / batch_size #/ dataset.N_CLASSES
                consistency_loss = consistency_weight * consistency_dist  
                consistency_relation_dist = torch.sum(losses.co_attention_based_relation_mse_loss(activations, ema_activations)) / batch_size
                consistency_relation_loss = consistency_weight * consistency_relation_dist * args.consistency_relation_weight

            else:
                consistency_loss = 0.0
                consistency_relation_loss = 0.0
                consistency_weight = 0.0
                consistency_dist = 0.0
             #+ consistency_loss
                
            ## calculate the loss
            loss_classification = loss_fn(outputs[:labeled_bs], label_batch[:labeled_bs])
            if args.bt_first : 
                loss = loss_classification + bt_loss # supervised loss + self-supervised loss
            else : 
                loss = loss_classification
            # loss = loss_classification
            
            if (epoch > args.apply_all_loss) and (args.ema_consistency == 1):
                # a proposed method is applied
                if args.bt_loss : 
                    loss = loss_classification + consistency_loss + consistency_relation_loss + bt_loss 
                else : 
                    loss = loss_classification + consistency_loss + consistency_relation_loss 
                
                # supervised loss + 2 network matching loss + 2 network matching loss 2 + self-supervised loss 

            # lr = adjust_learning_rate(args, optimizer, train_dataloader, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # outputs_soft = F.softmax(outputs, dim=1)
            meters_loss.update(loss=loss)
            meters_loss_classification.update(loss=loss_classification)
            meters_loss_consistency.update(loss=consistency_loss)
            meters_loss_consistency_relation.update(loss=consistency_relation_loss)
            # meters_loss_bt.update(loss=bt_loss)

            iter_num = iter_num + 1

            # write tensorboard
            if i % 100 == 0:
                writer.add_scalar('lr', lr_, iter_num)
                # writer.add_scalar('lr', lr, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_classification', loss_classification, iter_num)
                writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
                
                if args.bt_loss : 

                    logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, consistency loss: {:.6f}, consistency relation loss: {:.6f}, barlow twins loss: {:.6f}, consistency weight: {:.6f}, lr: {}"
                                .format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg, meters_loss_consistency.loss.avg, meters_loss_consistency_relation.loss.avg, meters_loss_bt.loss.avg, consistency_weight, optimizer.param_groups[0]['lr']))
                else : 
                    logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, consistency loss: {:.6f}, consistency relation loss: {:.6f}, consistency weight: {:.6f}, lr: {}"
                                .format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg, meters_loss_consistency.loss.avg, meters_loss_consistency_relation.loss.avg, consistency_weight, optimizer.param_groups[0]['lr']))

                image = inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('raw/Image', grid_image, iter_num)

                image = ema_inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('noise/Image', grid_image, iter_num)

        timestamp = get_timestamp()

        # validate student
        # 
        
        

        AUROCs, Accus, Senss, Specs = epochVal_metrics(model, val_dataloader)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()

        logging.info("\nVAL Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nVAL AUROC: {:6f}, VAL Accus: {:6f}, VAL Senss: {:6f}, VAL Specs: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))
        
        # test student
        # 
        AUROCs, Accus, Senss, Specs = epochVal_metrics(model, test_dataloader)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()

        logging.info("\nTEST Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))

        # save model
        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
        torch.save({    'epoch': epoch + 1,
                        'global_step': iter_num,
                        'state_dict': model.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'epochs'    : epoch
                        # 'AUROC'     : AUROC_best,
                   }
                   , save_mode_path
        )
        logging.info("save model to {}".format(save_mode_path))
        
        

        # update learning rate
        # lr_ = lr_ * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(iter_num+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
