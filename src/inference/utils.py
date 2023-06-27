import argparse
import errno
import math
import os
import os.path as osp

import pretrainedmodels
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in pretrainedmodels.__dict__ if name.islower() and not name.startswith("__"))


def get_train_args(passed_args=None):
    parser = argparse.ArgumentParser(description='PyTorch ResNet Model for Sketchy mAP Testing')
    parser.add_argument('--dataset',
                        # default="sketchy",
                        default="tuberlin",
                        help='dataset name')
    parser.add_argument('--root_dir', metavar='DIR',
                        default=r'E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\TUBerlin',
                        # default=r'E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Sketchy',
                        # default=r'/home/archlab/Datasets/dataset/Sketchy',
                        # default=r'/home/syp_pyCharm/Datasets/dataset/Sketchy',
                        help='path to dataset dir')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: se_resnet50)')
    parser.add_argument('--num_classes', metavar='N', type=int,
                        # default=100,
                        default=220,
                        help='number of classes (default: 100)')
    parser.add_argument('--num_hashing', metavar='N', type=int,
                        default=64,
                        # default=512,
                        help='number of hashing dimension (default: 64)')
    parser.add_argument('--batch_size',
                        default=4,
                        type=int, metavar='N', help='number of samples per batch')
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')

    parser.add_argument('--resume_dir',
                        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], '..','..', "cse_resnet50",
                                             "checkpoint"),
                        type=str, metavar='PATH', help='dir of model checkpoint (default: none)')
    parser.add_argument('--resume_file',
                        default='model_best.pth.tar',
                        type=str, metavar='PATH', help='file name of model checkpoint (default: none)')

    parser.add_argument('--ems-loss', dest='ems_loss', action='store_true',
                        help='use ems loss for the training')
    parser.add_argument('--precision', action='store_true', default=True,  help='report precision@100')
    parser.add_argument('--recompute', action='store_true', default=True,  help='recompute image feature')
    parser.add_argument('--visualize', action='store_true', default=True,  help='result save as image')
    parser.add_argument('--itq', action='store_true', help='compute binary code')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--zero_version', metavar='VERSION',
                        # default='zeroshot1',
                        default='zeroshot',
                        type=str,
                        help='zeroshot version for training and testing (default: zeroshot1)')

    parser.add_argument('--kd_lambda', metavar='LAMBDA', default='1.0', type=float,
                        help='lambda for kd loss (default: 1)')
    parser.add_argument('--kdneg_lambda', metavar='LAMBDA', default='0.3', type=float,
                        help='lambda for semantic adjustment (default: 0.3)')
    parser.add_argument('--sake_lambda', metavar='LAMBDA', default='1.0', type=float,
                        help='lambda for total SAKE loss (default: 1)')

    parser.add_argument('--contrastive_lambda', metavar='LAMBDA', default='0.1', type=float,
                        help='lambda for contrastive loss')
    parser.add_argument('--lfcm_lambda', metavar='LAMBDA', default='1.0', type=float,
                        help='lambda for contrastive loss')

    parser.add_argument('--temperature', metavar='LAMBDA', default='0.07', type=float,
                        help='lambda for temperature in contrastive learning')
    parser.add_argument('--contrastive_dim', metavar='N', type=int, default=128,
                        help='the dimension of contrastive feature (default: 128)')
    parser.add_argument('--topk', metavar='N', type=int, default=10,
                        help='save topk embeddings in memory bank (default: 10)')
    parser.add_argument('--memory_lambda', metavar='LAMBDA', default='1.0', type=float,
                        help='lambda for contrastive loss')

    if passed_args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(passed_args)


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # if is_best:
    #     filepath = '/'.join(filename.split('/')[0:-1])
    #     shutil.copyfile(filename, os.path.join(filepath,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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


def resume_from_checkpoint(model, path):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location='cpu')
        # args.start_epoch = checkpoint['epoch']

        save_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
        print('trashed vars from resume dict:')
        print(trash_vars)

        resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}

        model_dict.update(resume_dict)
        print(model.load_state_dict(model_dict, strict=False))

        # model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
        # return


def adjust_learning_rate(epoch, warmup_epochs=3, mode='exp', max_epoch=20, fix_epochs=0):
    lr = 1
    assert fix_epochs < max_epoch - warmup_epochs
    epoch -= fix_epochs
    max_epoch -= fix_epochs
    if epoch < 0:
        lr = lr  # fix base
    elif epoch < warmup_epochs:
        lr = lr * (0.01 + epoch / warmup_epochs)  # warmup
    elif epoch >= warmup_epochs * 2:
        if mode == 'cos':
            lr = lr * (1 + math.cos(math.pi * (epoch - 1 * warmup_epochs) / (max_epoch - 1 * warmup_epochs))) / 2
        elif mode == 'exp':
            lr = lr * math.pow(0.001, float(epoch - warmup_epochs) / (max_epoch - warmup_epochs))  # exp decay
        else:  # if mode == 'const':
            pass
    else:
        pass

    return lr


def fix_base_para(model):
    model.requires_grad_(True)
    print('training with base paramaters fixed')
    base_params, new_params = get_new_params(model)
    for para in base_params:
        para.requires_grad_(False)


def get_new_params(model, verbose=False, ignored_para=('second',)):
    base_params = []
    new_params = []
    for name, para in model.named_parameters():
        # if name contains ignored words, ignore this parameter.
        # if len(ignored_para) > 0 and sum([ig in name for ig in ignored_para]) > 0:
        #     continue

        if ('fc_tag' in name or 'linear' in name) and 'last' not in name:  # new parameters: CSE_fc, classifier
            if verbose:
                print(name)
            new_params.append(para)
        else:
            base_params.append(para)

    return base_params, new_params
