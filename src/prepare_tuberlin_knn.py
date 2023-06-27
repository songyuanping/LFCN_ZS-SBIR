import argparse
import datetime
import math
import os
import pickle
import sys
import time
from apex import amp
import itertools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import numpy as np
import pretrainedmodels
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ResnetModel import CSEResnetModel_KDHashing
from Sketchy import SketchImagePairedDataset
from TUBerlin import TUBerlinDataset
from senet import cse_resnet50
from tool import SupConLoss, MemoryStore, AverageMeter, validate_paired
from losses.network import AdversarialNetwork, ResNetFc, HDA_UDA, calc_coeff
from losses.loss import WeightedCrossMatchingTripletLoss
from losses.Smooth_AP import TripletLoss
from losses.logger import Logger
from losses.custom_loss import get_num_list, regularizer
from models.matcher import Matcher_Loss

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch CSE_ResNet Model for TUBerlin Training')
parser.add_argument('--root_dir', metavar='DIR',
                    default=r'E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\TUBerlin',
                    # default=r'/home/archlab/Datasets/dataset/TUBerlin',
                    # default=r'/home/syp_pyCharm/Datasets/dataset/TUBerlin',
                    help='path to dataset dir')
parser.add_argument('--savedir', '-s', metavar='DIR',
                    default=os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', "cse_resnet50",
                                         "checkpoint"), help='path to save dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cse_resnet50)')
parser.add_argument('--num_classes', metavar='N', type=int, default=220,
                    help='number of classes (default: 220)')
parser.add_argument('--num_hashing', metavar='N', type=int, default=64,
                    help='number of hashing dimension (default: 64)')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size',
                    default=120,
                    # default=130,  # archlab
                    type=int, metavar='N', help='number of samples per batch')
parser.add_argument('--lr', '--learning-rate',
                    default=0.0001,
                    type=float, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--fixbase_epochs',
                    default=3,
                    type=int, metavar='fixed epoches', help='fixed backbone for epoches')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-f', '--freeze_features', dest='freeze_features', action='store_true',
                    help='freeze features of the base network')
parser.add_argument('--ems-loss', dest='ems_loss', action='store_true',
                    help='use ems loss for the training')
parser.add_argument('--kd_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for kd loss (default: 1)')
parser.add_argument('--kdneg_lambda', metavar='LAMBDA', default='0.3', type=float,
                    help='lambda for semantic adjustment (default: 0.3)')
parser.add_argument('--sake_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for total SAKE loss (default: 1)')
parser.add_argument('--margin', metavar='LAMBDA', default='0.2', type=float,
                    help='lambda for total tri loss (default: 1)')
parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot', type=str,
                    help='zeroshot version for training and testing (default: zeroshot)')

parser.add_argument('--contrastive_lambda', metavar='LAMBDA', default='0.1', type=float,
                    help='lambda for contrastive loss')
parser.add_argument('--temperature', metavar='LAMBDA', default='0.07', type=float,
                    help='lambda for temperature in contrastive learning')
parser.add_argument('--contrastive_dim', metavar='N', type=int, default=128,
                    help='the dimension of contrastive feature (default: 128)')
parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                    help='resume from the latest epoch')
parser.add_argument('--topk', metavar='N', type=int, default=10,
                    help='save topk embeddings in memory bank (default: 10)')
parser.add_argument('--memory_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for contrastive loss')
parser.add_argument('--cls_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for cross entropy loss (default: 1)')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

global args
args = parser.parse_args()
model_t = cse_resnet50()
# load data
immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
imstd = [0.229, 0.224, 0.225]

transformations = transforms.Compose([transforms.ToPILImage(),

                                      # 原版本
                                      transforms.Resize([224, 224]),
                                      # 改进
                                      # transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize(immean, imstd)])

contrastive_transform = transforms.Compose([transforms.ToPILImage(),
                                            # 原版本
                                            transforms.Resize([224, 224]),
                                            # 改进
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                                                   p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize(immean, imstd)])

tuberlin_train = TUBerlinDataset(root_dir=args.root_dir, split='train', zero_version=args.zero_version,
                                 transform=transformations)
train_loader = DataLoader(dataset=tuberlin_train, batch_size=args.batch_size, shuffle=False, drop_last=False)
print("len(train_dataset):", len(tuberlin_train))

tuberlin_train_ext = TUBerlinDataset(root_dir=args.root_dir, split='train', zero_version=args.zero_version,
                                     version='ImageResized_ready', transform=transformations)
train_loader_ext = DataLoader(dataset=tuberlin_train_ext, batch_size=args.batch_size, shuffle=False, drop_last=False)
print("len(train_dataset):", len(tuberlin_train_ext))

if __name__ == '__main__':
    # with torch.no_grad():
    #     full_model = cse_resnet50(num_classes=1000).cuda()
    #     full_model.eval()
    #     # x = torch.rand((128, 3, 224, 224))
    #     #
    #     # print(full_model.features(x, torch.ones((128, 1))).shape)  # torch.Size([128, 2048, 7, 7])
    #     # print(full_model.logits(full_model.features(x, torch.ones((128, 1)))).shape)  # torch.Size([128, 1000])
    #     logits_list = []
    #     for i, (image, label_i) in enumerate(train_loader_ext):
    #         image, label_i = image.cuda(), torch.cat([label_i]).cuda()
    #         tag_ones = torch.ones(image.size()[0], 1).cuda()
    #         logits = full_model.features(image, tag_ones)
    #         logits = F.adaptive_avg_pool2d(logits, output_size=1)
    #         logits = logits.squeeze()
    #         logits_list.append(logits)
    #     logits_gallery = F.normalize(torch.cat(logits_list), dim=-1)
    #     print("logits_gallery.shape:", logits_gallery.shape)
    #     knn_list = []
    #     for i, (image, label_i) in enumerate(train_loader_ext):
    #         image, label_i = image.cuda(), torch.cat([label_i]).cuda()
    #         tag_ones = torch.ones(image.size()[0], 1).cuda()
    #         logits = full_model.features(image, tag_ones)
    #         logits = F.adaptive_avg_pool2d(logits, output_size=1)
    #         logits = logits.squeeze()
    #         logits = F.normalize(logits, dim=-1)
    #         sims = logits @ logits_gallery.T
    #         top100_score, top100_indexes = torch.topk(sims, dim=-1, k=100, largest=True)
    #         print("top100_score:", top100_score, "top100_indexes:", top100_indexes)
    #         knn_list.append(top100_indexes)
    #     knn_list = torch.cat(knn_list)
    #     print("knn_list.shape:", knn_list.shape)
    #     with open(os.path.join(BASE_DIR, "pretrained_tuberlin_images_nnidxs.pkl"), 'wb') as f:
    #         pickle.dump(knn_list.cpu().numpy(), f)
    #     with open(os.path.join(BASE_DIR, "pretrained_tuberlin_images_nnidxs.pkl"), 'rb') as f:
    #         knn_list = pickle.load(f)
    #     print("after load knn_list.shape:", knn_list.shape)

    with torch.no_grad():
        full_model = cse_resnet50(num_classes=1000).cuda()
        full_model.eval()
        # x = torch.rand((128, 3, 224, 224))
        #
        # print(full_model.features(x, torch.ones((128, 1))).shape)  # torch.Size([128, 2048, 7, 7])
        # print(full_model.logits(full_model.features(x, torch.ones((128, 1)))).shape)  # torch.Size([128, 1000])
        logits_list = []
        for i, (sketch, label_s) in enumerate(train_loader):
            sketch, label_s = sketch.cuda(), torch.cat([label_s]).cuda()
            tag_zeros = torch.zeros(sketch.size()[0], 1).cuda()
            logits = full_model.features(sketch, tag_zeros)
            logits = F.adaptive_avg_pool2d(logits, output_size=1)
            logits = logits.squeeze()
            logits_list.append(logits)
        logits_gallery = F.normalize(torch.cat(logits_list), dim=-1)
        print("sketch logits_gallery.shape:", logits_gallery.shape)
        knn_list = []
        for i, (sketch, label_s) in enumerate(train_loader):
            sketch, label_s = sketch.cuda(), torch.cat([label_s]).cuda()
            tag_zeros = torch.zeros(sketch.size()[0], 1).cuda()
            logits = full_model.features(sketch, tag_zeros)
            logits = F.adaptive_avg_pool2d(logits, output_size=1)
            logits = logits.squeeze()
            logits = F.normalize(logits, dim=-1)
            sims = logits @ logits_gallery.T
            top100_score, top100_indexes = torch.topk(sims, dim=-1, k=100, largest=True)
            print("top100_score:", top100_score, "top100_indexes:", top100_indexes)
            knn_list.append(top100_indexes)
        knn_list = torch.cat(knn_list)
        print("knn_list.shape:", knn_list.shape)
        with open(os.path.join(BASE_DIR, "../pretrained_tuberlin_sketches_nnidxs.pkl"), 'wb') as f:
            pickle.dump(knn_list.cpu().numpy(), f)
        with open(os.path.join(BASE_DIR, "../pretrained_tuberlin_sketches_nnidxs.pkl"), 'rb') as f:
            knn_list = pickle.load(f)
        print("after load knn_list.shape:", knn_list.shape)
