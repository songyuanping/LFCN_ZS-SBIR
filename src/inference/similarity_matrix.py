import os
import sys
import itertools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from src.Sketchy import SketchyDataset
from src.TUBerlin import TUBerlinDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from src.ResnetModel import CSEResnetModel_KDHashing
from utils import resume_from_checkpoint
from utils import get_train_args

import seaborn as sns
import matplotlib.pyplot as plt

from utils import mkdir_if_missing

mkdir_if_missing(os.path.join(BASE_DIR, "plots"))


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)

    plt.figure(dpi=1200)

    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=90)
    plt.xticks([])
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    # plt.xlim( -0.5,len(classes) - 0.5)
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()
    # plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "plots", "similarity_matrix.png"))
    plt.show()


def gen_similarity_matrix(args, resume_dir, get_precision=False, model=None, recompute=False, visualize=False):
    args.resume_dir = resume_dir
    print('prepare SBIR features using saved model')
    predicted_features_gallery, gt_labels_gallery, predicted_features_query, \
    gt_labels_query = prepare_features(args, model, args.itq)
    print("predicted_features_gallery.shape:", predicted_features_gallery.shape, "gt_labels_gallery.shape:",
          gt_labels_gallery.shape, "predicted_features_query.shape:", predicted_features_query.shape,
          "gt_labels_query.shape:", gt_labels_query.shape)

    mean_sketch_fea, mean_image_fea = [], []
    for label in sorted(np.unique(gt_labels_gallery)):
        print("label:", label)
        sketch_index = np.where(gt_labels_query == label)
        print("sketch_index:", sketch_index)
        print("np.mean(predicted_features_query[sketch_index], axis=0):",
              np.mean(predicted_features_query[sketch_index], axis=0).shape)
        mean_sketch_fea.append(np.mean(predicted_features_query[sketch_index], axis=0))

        image_index = np.where(gt_labels_gallery == label)
        mean_image_fea.append(np.mean(predicted_features_gallery[image_index], axis=0))

    mean_sketch_fea = np.array(mean_sketch_fea)
    mean_image_fea = np.array(mean_image_fea)
    print("mean_sketch_fea.shape:", mean_sketch_fea.shape, "mean_image_fea.shape:", mean_image_fea.shape)

    # sns.set()
    # f, ax = plt.subplots()

    similarity_matrix = mean_sketch_fea @ mean_image_fea.T
    # print(similarity_matrix.shape, similarity_matrix)  # 打印出来看看
    # sns.heatmap(similarity_matrix, ax=ax)  # 画热力图
    # plt.yticks(ticks=np.arange(30), labels=)

    cnf_matrix = similarity_matrix
    # attack_types = ["banana",
    #                 "bus",
    #                 "tractor",
    #                 "suitcase",
    #                 "streetlight",
    #                 "telephone",
    #                 "bottle opener",
    #                 "canoe",
    #                 "fan",
    #                 "teacup",
    #                 "penguin",
    #                 "laptop",
    #                 "shoe",
    #                 "lighter",
    #                 "hot air balloon",
    #                 "pizza",
    #                 "brain",
    #                 "ant",
    #                 "t-shirt",
    #                 "trombone",
    #                 "windmill",
    #                 "snowboard",
    #                 "table",
    #                 "rollerblades",
    #                 "parachute",
    #                 "space shuttle",
    #                 "bridge",
    #                 "frying-pan",
    #                 "bread",
    #                 "horse"]
    attack_types = [
        "cup",
        "swan",
        "harp",
        "squirrel",
        "snail",
        "ray",
        "pineapple",
        "volcano",
        "rifle",
        "scissors",
        "parrot",
        "windmill",
        "teddy_bear",
        "tree",
        "wine_bottle",
        "deer",
        "chicken",
        "airplane",
        "wheelchair",
        "tank",
        "umbrella",
        "butterfly",
        "camel",
        "horse",
        "bell"]
    plot_confusion_matrix(cnf_matrix, classes=attack_types, title=' ')


def prepare_features(args, model=None, itq=False):
    # load data
    immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(immean, imstd)
    ])

    if args.dataset == 'tuberlin':
        args.num_classes = 220
        test_zero_ext = TUBerlinDataset(root_dir=args.root_dir, split='zero', version='ImageResized_ready',
                                        zero_version=args.zero_version, transform=transformations, aug=False,
                                        first_n_debug=1000000)
        test_zero = TUBerlinDataset(root_dir=args.root_dir, split='zero', zero_version=args.zero_version,
                                    transform=transformations, aug=False, first_n_debug=1000000)

    elif args.dataset == 'sketchy':
        if args.zero_version == 'zeroshot2':
            args.num_classes = 104
        else:
            args.zero_version = 'zeroshot1'
            args.num_classes = 100
        test_zero_ext = SketchyDataset(root_dir=args.root_dir, split='zero', version='all_photo',
                                       zero_version=args.zero_version, transform=transformations, aug=False)
        test_zero = SketchyDataset(root_dir=args.root_dir, split='zero', zero_version=args.zero_version,
                                   transform=transformations, aug=False)

    else:
        print('not support dataset', args.dataset)

    datasets = [test_zero.file_ls, test_zero.labels, test_zero_ext.file_ls, test_zero_ext.labels]

    zero_loader_ext = DataLoader(dataset=test_zero_ext, batch_size=args.batch_size, shuffle=False, num_workers=1)

    zero_loader = DataLoader(dataset=test_zero, batch_size=args.batch_size, shuffle=False, num_workers=1)

    print(str(datetime.datetime.now()) + ' data loaded.')

    if model is None:
        model = CSEResnetModel_KDHashing(args.arch, args.num_hashing, args.num_classes)
        model = nn.DataParallel(model).cuda()
        print(str(datetime.datetime.now()) + ' model inited.')

        # resume from a checkpoint
        if args.resume_file:
            resume = os.path.join(args.resume_dir, args.resume_file)
        else:
            resume = os.path.join(args.resume_dir, 'checkpoint.pth.tar')

        resume_from_checkpoint(model, resume)

    cudnn.benchmark = True

    predicted_features_gallery, gt_labels_gallery = get_features(zero_loader_ext, model, args.pretrained)

    predicted_features_query, gt_labels_query = get_features(zero_loader, model, args.pretrained, 0)

    return predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query


@torch.no_grad()
def get_features(data_loader, model, pretrained=False, tag=1):
    # switch to evaluate mode
    model.eval()
    features_all = []
    targets_all = []
    avgpool = nn.AdaptiveAvgPool2d(1).cuda()
    for i, (input, target) in enumerate(data_loader):
        if i % 10 == 0:
            print(i, end=' ', flush=True)

        tag_input = (torch.ones(input.size()[0], 1) * tag).cuda()
        input = torch.autograd.Variable(input, requires_grad=False).cuda()

        # compute output
        # features = avgpool(model.module.features(input, tag_input)).cpu().detach().numpy()
        features = model.module.original_model.features(input, tag_input)
        if pretrained:
            features = model.module.original_model.avg_pool(features)
            features = features.view(features.size(0), -1)
        else:
            features, _ = model.module.original_model.hashing(features)

        features = F.normalize(features)
        features = features.cpu().detach().numpy()

        features_all.append(features.reshape(input.size()[0], -1))
        targets_all.append(target.detach().numpy())

    print('')

    features_all = np.concatenate(features_all)
    targets_all = np.concatenate(targets_all)

    print('Features ready: {}, {}'.format(features_all.shape, targets_all.shape))

    return features_all, targets_all


def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap


def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top


if __name__ == '__main__':
    args = get_train_args()
    if args.dataset == 'sketchy2':
        args.dataset = 'sketchy'
        args.zero_version = 'zeroshot2'
    savedir = f'{args.dataset}_kd({args.kd_lambda})_kdneg({args.kdneg_lambda})_sake({args.sake_lambda})_' \
              f'dim({args.num_hashing})_lfcm({args.lfcm_lambda})_' \
              f'contrastive({args.contrastive_dim}-{args.contrastive_lambda})_T({args.temperature})_' \
              f'memory({args.topk}-{args.memory_lambda})'
    args.resume_dir = os.path.join(args.resume_dir, savedir)
    # sys.stdout = Logger(os.path.join(args.resume_dir, "visualize_" + time.strftime('%Y-%m-%d-%H-%M-%S') + '-log.txt'))

    gen_similarity_matrix(args, args.resume_dir, get_precision=args.precision, recompute=args.recompute,
                          visualize=args.visualize)
