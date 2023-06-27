import argparse
import datetime
import math
import os
import sys
import pickle
import time
import itertools
from apex import amp

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
# from train_cse_resnet_tuberlin_ext import validate, accuracy, save_checkpoint, adjust_learning_rate

from ResnetModel import CSEResnetModel_KDHashing, GCN
from Sketchy import SketchyDataset, SketchImagePairedDataset
from senet import cse_resnet50
from tool import SupConLoss, MemoryStore, AverageMeter, validate_paired
from train_cse_resnet_tuberlin_ext import SoftCrossEntropy, worker_init_fn_seed, fix_base_para
from losses.logger import Logger
from models.matcher import Matcher_Loss

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch CSE_ResNet Model for Sketchy Training')
parser.add_argument('--root_dir', metavar='DIR',
                    # default=r'D:\ypsong\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Sketchy',
                    default=r'/home/archlab/Datasets/dataset/Sketchy',
                    # default=r'/home/syp_pyCharm/Datasets/dataset/Sketchy',
                    help='path to dataset dir')
parser.add_argument('--savedir', '-s', metavar='DIR',
                    default=os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', "cse_resnet50",
                                         "checkpoint"),
                    help='path to save model dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: cse_resnet50)')
parser.add_argument('--num_classes', metavar='N', type=int, default=100,
                    help='number of classes (default: 100)')
parser.add_argument('--num_hashing', metavar='N', type=int,
                    # default=64,
                    default=512,
                    help='number of hashing dimension (default: 64)')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size',
                    default=114,
                    type=int, metavar='N', help='number of samples per batch')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--fixbase_epochs',
                    default=3,
                    type=int, metavar='fixed epoches', help='fixed backbone for epoches')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 50)')
# gat_config
parser.add_argument('--gc_in_channel',
                    default=300,  # archlab
                    type=int, metavar='N', help='number of samples per batch')
parser.add_argument('--adj_files', metavar='DIR',
                    default=[os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', 'data', "Sketchy", file)
                             for file in
                             ['glove-wiki_cosine_adj.pkl_emb', 'word2vec_cosine_adj.pkl_emb']],
                    help='path to class adj file dirs')
parser.add_argument('--word_vec_path', metavar='DIR', type=str,
                    default=os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', 'data', "Sketchy",
                                         "train_word_feature.pkl"), help='path to word vector file dir')
parser.add_argument('--weight_threshold',
                    default=0.4,
                    type=float, help='weight threshold in create graph')
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
parser.add_argument('--kdneg_lambda2', metavar='LAMBDA', default='0.1', type=float,
                    help='lambda for semantic adjustment (default: 0.1)')
parser.add_argument('--sake_lambda', metavar='LAMBDA',
                    # default='1.0',
                    default='0.0',
                    type=float,
                    help='lambda for total SAKE loss (default: 1)')
parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot1', type=str,
                    help='zeroshot version for training and testing (default: zeroshot1)')

parser.add_argument('--contrastive_lambda', metavar='LAMBDA', default='0.1', type=float,
                    help='lambda for contrastive loss')
parser.add_argument('--lfcm_lambda', metavar='LAMBDA',
                    # default='1.0',
                    default='0.0',
                    type=float,
                    help='lambda for contrastive loss')
parser.add_argument('--temperature', metavar='LAMBDA', default='0.07', type=float,
                    help='lambda for temperature in contrastive learning')
parser.add_argument('--contrastive_dim', metavar='N', type=int, default=128,
                    help='the dimension of contrastive feature (default: 128)')
parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                    help='resume from the latest epoch')
parser.add_argument('--topk', metavar='N', type=int, default=10,
                    help='save topk embeddings in memory bank (default: 10)')
parser.add_argument('--memory_lambda', metavar='LAMBDA',
                    # default='1.0',
                    default='0.0',
                    type=float,
                    help='lambda for contrastive loss')
parser.add_argument('--cls_lambda', metavar='LAMBDA',
                    default='1.0',
                    # default='0.0',
                    type=float,
                    help='lambda for cross entropy loss (default: 1)')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EMSLoss(nn.Module):
    def __init__(self, m=4):
        super(EMSLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.m = m

    def forward(self, inputs, targets):
        mmatrix = torch.ones_like(inputs)
        for ii in range(inputs.size()[0]):
            mmatrix[ii, int(targets[ii])] = self.m

        inputs_m = torch.mul(inputs, mmatrix)
        return self.criterion(inputs_m, targets)


def main():
    global args
    args = parser.parse_args()
    if args.zero_version == 'zeroshot2':
        args.num_classes = 104
    print("args.ems_loss:", args.ems_loss)

    print("args.freeze_features:", args.freeze_features)
    # create model
    model = CSEResnetModel_KDHashing(args.arch, args.num_hashing, args.num_classes,
                                     freeze_features=args.freeze_features, ems=args.ems_loss, module='CSE')
    gcn_model = GCN(num_classes=args.num_classes, gc_in_channel=args.gc_in_channel, t=args.weight_threshold,
                    adj_files=args.adj_files)
    model, gcn_model = model.cuda(), gcn_model.cuda()
    # model = nn.DataParallel(model).cuda()

    print(str(datetime.datetime.now()) + ' student model inited.')
    model_t = cse_resnet50()
    model_t = nn.DataParallel(model_t).cuda()
    print(str(datetime.datetime.now()) + ' teacher model inited.')

    # define loss function (criterion) and optimizer
    if args.ems_loss:
        print("**************  Use EMS Loss!")
        curr_m = 1
        criterion_train = EMSLoss(curr_m).cuda()
    else:
        criterion_train = nn.CrossEntropyLoss().cuda()

    criterion_contrastive = SupConLoss(args.temperature).cuda()
    criterion_train_kd = SoftCrossEntropy().cuda()
    criterion_test = nn.CrossEntropyLoss().cuda()
    criterion_tri = Matcher_Loss().cuda()

    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), criterion_tri.parameters(), gcn_model.parameters()), lr=args.lr,
        weight_decay=args.weight_decay)
    [model, criterion_tri, gcn_model], optimizer = amp.initialize([model, criterion_tri, gcn_model], optimizer)
    model = nn.DataParallel(model)

    cudnn.benchmark = True

    sketch_memory = MemoryStore(args.num_classes, args.topk, 2048)

    # load data
    immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]

    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([224, 224]),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(immean, imstd)])

    contrastive_transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize([224, 224]),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                                                       p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                transforms.ToTensor(),
                                                transforms.Normalize(immean, imstd)])

    sketchy_train = SketchyDataset(root_dir=args.root_dir, split='train', zero_version=args.zero_version,
                                   transform=transformations, aug=True, cid_mask=True,
                                   contrastive_transform=contrastive_transform)
    train_loader = DataLoader(dataset=sketchy_train, batch_size=int(args.batch_size / 11 * 5), shuffle=True,
                              num_workers=3,
                              worker_init_fn=worker_init_fn_seed)

    sketchy_train_ext = SketchyDataset(root_dir=args.root_dir, split='train', version='all_photo',
                                       zero_version=args.zero_version, transform=transformations, aug=True,
                                       cid_mask=True, contrastive_transform=contrastive_transform)

    train_loader_ext = DataLoader(dataset=sketchy_train_ext, batch_size=args.batch_size - int(args.batch_size / 11 * 5),
                                  shuffle=True, num_workers=3, worker_init_fn=worker_init_fn_seed)

    sketchy_val = SketchImagePairedDataset(root_dir=args.root_dir, zero_version=args.zero_version, shuffle=True,
                                           transform=transformations)
    val_loader = DataLoader(dataset=sketchy_val, batch_size=args.batch_size // 2, shuffle=True, num_workers=1,
                            drop_last=True)
    ########
    with open(args.word_vec_path, 'rb') as f:
        vec_matrix = torch.tensor(pickle.load(f)).cuda()
    ########

    print(str(datetime.datetime.now()) + ' data loaded.')

    if args.evaluate:
        acc1 = validate(val_loader, model, criterion_test, criterion_train_kd, model_t)
        print('Acc is {.4f}'.format(acc1))
        return

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    savedir = f'sketchy_kd({args.kd_lambda})_kdneg({args.kdneg_lambda})_sake({args.sake_lambda})_' \
              f'cls({args.cls_lambda})_dim({args.num_hashing})_lfcm({args.lfcm_lambda})_' \
              f'contrastive({args.contrastive_dim}-{args.contrastive_lambda})_T({args.temperature})_' \
              f'memory({args.topk}-{args.memory_lambda})'

    if not os.path.exists(os.path.join(args.savedir, savedir)):
        os.makedirs(os.path.join(args.savedir, savedir))

    sys.stdout = Logger(os.path.join(args.savedir, savedir, time.strftime('train-%Y-%m-%d-%H-%M-%S') + '-log.txt'))
    print('model:', model)
    print(time.strftime('train-%Y-%m-%d-%H-%M-%S'))
    print(args)
    # model.load_state_dict(torch.load(os.path.join(args.savedir, savedir, "epoch_2_model_best.pth.tar"))["state_dict"])
    # dic={}
    # for now_keys,values in zip(model.state_dict().keys(),torch.load(os.path.join(args.savedir, savedir, "model_best.pth.tar"))["state_dict"].values()):
    #     dic[now_keys]=values
    # model.load_state_dict(dic)
    print("load pretrained model state dict successfully!")

    best_acc1 = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        if epoch < args.fixbase_epochs:
            fix_base_para(model)
        else:
            model.requires_grad_(True)

        if args.ems_loss:
            if epoch in [20, 25]:
                new_m = curr_m * 2
                print("update m at epoch {}: from {} to {}".format(epoch, curr_m, new_m))
                criterion_train = EMSLoss(new_m).cuda()
                curr_m = new_m

        train(train_loader, train_loader_ext, model, criterion_train, criterion_train_kd, criterion_contrastive,
              optimizer, epoch, model_t, sketch_memory, criterion_tri, gcn_model, vec_matrix)
        acc1 = validate_paired(val_loader, model, criterion_test, criterion_train_kd, gcn_model, vec_matrix,
                               test_num=120)
        # val_accs.append(acc1.item())
        # print('Validated accuracy: {}'.format(val_accs))
        #
        # is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'gcn_state_dict': gcn_model.state_dict(),
            'tri_state_dict': criterion_tri.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, filename=os.path.join(args.savedir, savedir, f'epoch_{epoch}_model_best.pth.tar'))


def train(train_loader, train_loader_ext, model, criterion, criterion_kd, criterion_contrastive,
          optimizer, epoch, model_t, sketch_memory, criterion_tri, gcn_model, vec_matrix):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    losses_contrastive = AverageMeter()
    losses_memory = AverageMeter()
    losses_lfcm = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    criterion_tri.train()
    gcn_model.train()
    model_t.eval()
    sketch_memory.flush()
    end = time.time()
    print("len(train_loader): ", len(train_loader), "len(train_loader_ext): ", len(train_loader_ext))
    for i, ((sketch, sketch1, sketch2, label_s, sketch_cid_mask),
            (image, image1, image2, label_i, image_cid_mask)) in enumerate(zip(train_loader, train_loader_ext)):

        sketch, sketch1, sketch2, label_s, sketch_cid_mask, image, image1, image2, label_i, image_cid_mask = \
            sketch.cuda(), sketch1.cuda(), sketch2.cuda(), torch.cat([label_s]).cuda(), torch.cat(
                [sketch_cid_mask]).cuda(), \
            image.cuda(), image1.cuda(), image2.cuda(), torch.cat([label_i]).cuda(), torch.cat([image_cid_mask]).cuda()
        tag_zeros = torch.zeros(sketch.size()[0], 1)
        tag_ones = torch.ones(image.size()[0], 1)
        tag_all = torch.cat([tag_zeros, tag_ones], dim=0).cuda()

        sketch_shuffle_idx = torch.randperm(sketch.size(0))
        image_shuffle_idx = torch.randperm(image.size(0))
        sketch = sketch[sketch_shuffle_idx]
        sketch1 = sketch1[sketch_shuffle_idx]
        sketch2 = sketch2[sketch_shuffle_idx]
        label_s = label_s[sketch_shuffle_idx].type(torch.LongTensor).view(-1, )
        sketch_cid_mask = sketch_cid_mask[sketch_shuffle_idx].float()

        image = image[image_shuffle_idx]
        image1 = image1[image_shuffle_idx]
        image2 = image2[image_shuffle_idx]
        label_i = label_i[image_shuffle_idx].type(torch.LongTensor).view(-1, )
        image_cid_mask = image_cid_mask[image_shuffle_idx].float()

        target_all = torch.cat([label_s, label_i]).cuda()
        cid_mask_all = torch.cat([sketch_cid_mask, image_cid_mask]).cuda()

        output_kd, hash_code, features, features_map = model(
            torch.cat([sketch, image, sketch1, image1, sketch2, image2], 0),
            torch.cat([tag_all, tag_all, tag_all], 0))
        output = gcn_model(features, vec_matrix)
        output = output[:tag_all.size(0)]
        # output_kd = output_kd[:tag_all.size(0)]
        features = F.normalize(features, p=2, dim=-1)
        sketch_memory.add_entries(features[:tag_zeros.size(0)].detach(), label_s.detach(),
                                  features[tag_zeros.size(0):tag_all.size(0)].detach(), label_i.detach())
        loss_memory = sketch_memory.memory_loss(features[tag_zeros.size(0):tag_all.size(0)].detach(),
                                                label_i.detach())

        contrastive_feature = F.normalize(hash_code[tag_all.size(0):], p=2, dim=-1)

        contrastive_feature1 = torch.unsqueeze(contrastive_feature[:tag_all.size(0)], 1)
        contrastive_feature2 = torch.unsqueeze(contrastive_feature[tag_all.size(0):], 1)

        with torch.no_grad():
            output_t = model_t(torch.cat([sketch, image], 0), tag_all)
            output_t1 = model_t(torch.cat([sketch1, image1], 0), tag_all)
            output_t2 = model_t(torch.cat([sketch2, image2], 0), tag_all)

        loss = criterion(output, target_all)
        loss_contrastive = criterion_contrastive(torch.cat([contrastive_feature1, contrastive_feature2], 1), target_all)
        loss_lfcm = criterion_tri(F.normalize(hash_code, dim=-1), features_map,
                                  torch.cat([target_all, target_all, target_all]))
        # 计算图片经过提取后得到的损失
        loss_kd = criterion_kd(output_kd[:tag_all.size(0)], output_t * args.kd_lambda, tag_all,
                               cid_mask_all * args.kdneg_lambda)
        loss_kd += criterion_kd(output_kd[tag_all.size(0):], torch.cat([output_t1, output_t2]) * args.kd_lambda,
                                torch.cat([tag_all, tag_all]),
                                torch.cat([cid_mask_all, cid_mask_all]) * args.kdneg_lambda2)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target_all, topk=(1, 5))
        losses.update(loss.item(), tag_all.size(0))
        losses_kd.update(loss_kd.item(), tag_all.size(0))
        losses_contrastive.update(loss_contrastive.item(), tag_all.size(0) * 2)
        losses_memory.update(loss_memory.item(), tag_ones.size(0))
        losses_lfcm.update(loss_lfcm.item(), target_all.size(0))
        top1.update(acc1[0], tag_all.size(0))
        top5.update(acc5[0], tag_all.size(0))

        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss_total = args.cls_lambda * loss + args.sake_lambda * loss_kd + args.contrastive_lambda * loss_contrastive + \
                     args.memory_lambda * loss_memory + args.lfcm_lambda * loss_lfcm

        # loss_total.backward()
        with amp.scale_loss(loss_total, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == min(len(train_loader), len(train_loader_ext)) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_kd.val:.3f} {loss_contrastive.val:.3f} {loss_memory.val:.3f} {loss_tri.val:.3f}'
                  '({loss.avg:.3f} {loss_kd.avg:.3f} {loss_contrastive.avg:.3f} {loss_memory.avg:.3f} {loss_tri.avg:.3f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Memory used {used:.2f}%'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, loss_kd=losses_kd, loss_contrastive=losses_contrastive, loss_memory=losses_memory,
                loss_tri=losses_lfcm, top1=top1, used=sketch_memory.get_memory_used_percent()))


@torch.no_grad()
def validate(val_loader, model, criterion, criterion_kd, model_t):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_t.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = torch.autograd.Variable(input, requires_grad=False).cuda()
        target = target.type(torch.LongTensor).view(-1, )
        target = torch.autograd.Variable(target).cuda()

        # compute output
        # with torch.no_grad():
        output_t = model_t(input, torch.zeros(input.size()[0], 1).cuda())
        output, output_kd, _, __ = model(input, torch.zeros(input.size()[0], 1).cuda())

        loss = criterion(output, target)
        loss_kd = criterion_kd(output_kd, output_t)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_kd.update(loss_kd.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0 or i == len(val_loader) - 1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} {loss_kd.val:.3f} ({loss.avg:.3f} {loss_kd.avg:.3f})\t'
                  'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, loss_kd=losses_kd,
                top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch):
    if epoch < args.fixbase_epochs:  # fix pretrained para in first few epochs
        lr = args.lr * (1.0 + epoch) / args.fixbase_epochs
    else:
        # lr = args.lr * 0.5 * (1.0 + math.cos(float(epoch) / args.epochs * math.pi))
        lr = args.lr * math.pow(0.1, float(epoch) / args.epochs)
    print('epoch: {}, lr: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
