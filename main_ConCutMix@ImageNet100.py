import argparse
import os
import random
import shutil
import time
import warnings
import math
import neptune
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models_office
import csv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset.cifar import IMBALANCECIFAR10
from dataset.cifar import IMBALANCECIFAR100
from dataset.cifar import Cifar100
from dataset.imagenet import ImageNetLT, ImageNet100
from dataset.inaturalist import INaturalist
from dataset.PlacesLT import PlacesLT
from loss.contrastive import BalSCL
from loss.logitadjust import LogitAdjust, cutmix_cross_entropy
from models.resnet32 import BCLModel_32
from models.resnext import BCLModel
from utils import rand_augment_transform
from utils import shot_acc, GaussianBlur
from utils import CIFAR10Policy
from concurrent.futures import ThreadPoolExecutor
import sys
import logging
import tqdm
import json
import faiss


# 数据集配置
DATASET = "ImageNet100"
IMG_SIZE = (224, 224)
# ConCutMix 对比模型相关参数
ARCH = "resnet50"
NUM_CLASS = 100
FEAT_DIM = 1024
USE_NORM = True
CONCUTMIX_CKPT = "/home/Users/dqy/Projects/ConCutMix/log/baseline_ImageNet100_resnet50_batchsize_256_epochs_100_temp_0.07_cutmix_prob_0.5_topk_30_scaling_factor_200_255_tau_0.99_lr_0.1_sim-sim/ConCutMix_ckpt.best.pth.tar"
MEAN_CONTRAST = [0.485, 0.456, 0.406]
STD_CONTRAST = [0.229, 0.224, 0.225]
# ConCutMix 分类模型相关参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_model(arch):
    if arch == 'resnet50':
        model = BCLModel(name='resnet50', feat_dim=FEAT_DIM,
                         num_classes=NUM_CLASS,
                         use_norm=USE_NORM)
    elif arch == 'resnext50':
        model = BCLModel(name='resnext50', feat_dim=FEAT_DIM, num_classes=NUM_CLASS,
                         use_norm=USE_NORM)
    elif arch == 'resnet32':
        model = BCLModel_32(name='resnet32', feat_dim=FEAT_DIM,
                            num_classes=NUM_CLASS,
                            use_norm=USE_NORM)
    elif arch == "resnet152":
        model = BCLModel(name='resnet152', feat_dim=FEAT_DIM,
                         num_classes=NUM_CLASS,
                         use_norm=USE_NORM)
    elif arch == 'resnext101':
        model = BCLModel(name='resnext101', feat_dim=FEAT_DIM,
                         num_classes=NUM_CLASS,
                         use_norm=USE_NORM)
    else:
        raise NotImplementedError('This model is not supported')
    return model


def load_Contrast_model():
    model = get_model(ARCH)
    checkpoint = torch.load(CONCUTMIX_CKPT, map_location='cuda:0')
    new_dict = {}
    for k, v in checkpoint['state_dict'].items():
        new_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_dict)
    return model.cuda()


def retrieval_Faiss(Faiss_folder, query_feature, k=5):
    # 获取 Faiss 检索文件的路径
    index_path = os.path.join(Faiss_folder, "Contrast_index.faiss")
    source_path = os.path.join(Faiss_folder, "index_paths.npy")
    # 加载索引和路径映射
    # print("Loading Faiss database...")
    index = faiss.read_index(index_path)
    feature_paths = np.load(source_path, allow_pickle=True)
    # 迁移到 GPU
    # print("Converting Faiss database to GPU...")
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    # 对给定的特征进行查询
    faiss.normalize_L2(query_feature)
    # print("Retrieving...")
    D, I = index.search(query_feature, k)  # D=相似度, I=索引
    # 输出前 k 个相似图像的路径
    # for rank, idx in enumerate(I[0]):  # 这里的 [0] 表示对第 0 个查询向量的检索结果
    #     print(f"Top-{rank+1}: {feature_paths[idx]} (similarity={D[0][rank]:.4f})")
    return D, I


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # 减去最大值防止数值溢出
    return e_x / e_x.sum(axis=0)


def renormalize_images_simple(images: torch.Tensor) -> torch.Tensor:
    """使用全局变量的简化版本"""
    # 重塑全局变量以便广播
    original_mean = torch.tensor(MEAN, device=images.device).view(1, -1, 1, 1)
    original_std = torch.tensor(STD, device=images.device).view(1, -1, 1, 1)
    new_mean = torch.tensor(MEAN_CONTRAST, device=images.device).view(1, -1, 1, 1)
    new_std = torch.tensor(STD_CONTRAST, device=images.device).view(1, -1, 1, 1)
    
    # 反归一化然后重新归一化
    denormalized = images * original_std + original_mean
    renormalized = (denormalized - new_mean) / new_std
    
    return renormalized


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def name_to_label():
    root = f"/home/Users/dqy/Dataset/{DATASET}/format_ImageNet/images/train/"
    categories = sorted(os.listdir(root))
    mapping = {category: idx for idx, category in enumerate(categories)}
    return mapping


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet',
                    choices=['inat', 'imagenet', 'cifar10', 'cifar100', 'Places_LT', 'ImageNet100',
                             'ImageNet100-LT', 'Places365', 'Places365-LT', 'Cifar100', 'Cifar100-LT', 'Cifar10',
                             'Cifar10-LT', 'iNaturalist'])
parser.add_argument('--data', default='/DATACENTER/raid5/zjg/imagenet', metavar='DIR')
parser.add_argument('--arch', default='resnext50',
                    choices=['resnet50', 'resnext50', 'resnet32', 'resnet152', 'resnext101'])
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float, help='cross entropy loss weight')
parser.add_argument('--beta', default=0.35, type=float, help='supervised contrastive loss weight')
parser.add_argument('--randaug', default=True, type=bool, help='use RandAugmentation for classification branch')
parser.add_argument('--cl_views', default='sim-sim', type=str,
                    choices=['sim-sim', 'sim-rand', 'rand-rand', 'cutout-sim', 'none-sim', "cutout-none",
                             "uncutout-sim", "unauto-sim", "cutmix-sim", "cutoutmix-sim", "uncutout-cutmix_sim"],
                    help='Augmentation strategy for contrastive learning views')
parser.add_argument('--feat_dim', default=1024, type=int, help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--cos', default=False, action='store_true',
                    help='lr decays by cosine scheduler. ')
parser.add_argument('--use_norm', action='store_true',
                    help='cosine classifier.')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')
parser.add_argument('--imb_factor', default=1, type=float)
parser.add_argument('--grad_c', action='store_true', )
parser.add_argument('--file_name', default="", type=str)
parser.add_argument('--device_ids', default=[0, 1, 2, 3], type=int, nargs="*")
parser.add_argument('--save_epoch', default=None, type=int)
parser.add_argument('--auto_resume', action='store_true')
parser.add_argument('--reload_torch', default=None, type=str,
                    help='load supervised model from torchvision')
parser.add_argument('--num_classes', default=None, type=int,
                    help='num_classes')
# neptune
parser.add_argument('--logger', default="none", type=str, choices=["neptune", "none"])
parser.add_argument('--ne_token', default="", type=str)
parser.add_argument('--ne_project', default="", type=str)
parser.add_argument('--ne_run', default=None, type=str)

# ablation
parser.add_argument('--Background_sampler', default="uniform", type=str, choices=["balance", "reverse", "uniform"])
parser.add_argument('--Foreground_sampler', default="balance", type=str, choices=["reverse", "balance", "uniform"])

# cutmix:

parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')

# Contrastive CutMix
parser.add_argument('--l_d_warm', default=0, type=int)
parser.add_argument('--scaling_factor', default=[2, 256], nargs='*', type=int,
                    help='scaling_factor=[a,b]=a/b')
parser.add_argument('--tau', default=1, type=float)
parser.add_argument('--topk', default=1, type=int)

# feature extraction
parser.add_argument('--extract_feature', action='store_true', help="extract features and save to dir")
parser.add_argument('--extract_phase', default='val', help="train or val", choices=['train', 'val'])
parser.add_argument('--extract_type', default='contrast', help="contrast or classification", choices=['contrast', 'classification'])
parser.add_argument('--save_dir', default="", type=str)

# knowledge base
parser.add_argument('--knowledge_base', default="", type=str)
parser.add_argument('--prior_weight', default=0.0, type=float)
parser.add_argument('--retrieval_k', default=5, type=int)
parser.add_argument('--only_retrieval', action='store_true')
parser.add_argument('--result_json_path', default="", type=str)


def label_to_name(args):
    root = f"/home/Users/dqy/Dataset/{args.dataset}/format_ImageNet/images/train/"
    categories = sorted(os.listdir(root))
    mapping = {idx: category for idx, category in enumerate(categories)}
    return mapping


def main():
    args = parser.parse_args()
    if args.extract_feature:
        assert args.reload
        assert args.resume != ""
        assert args.save_dir != ""
    args.store_name = '_'.join(
        [args.file_name, args.dataset, args.arch, 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'temp',
         str(args.temp), "cutmix_prob", str(args.cutmix_prob), "topk", str(args.topk), "scaling_factor",
         str(args.scaling_factor[0]), str(args.scaling_factor[1]), "tau", str(args.tau)
            , 'lr', str(args.lr), args.cl_views])
    # print(args.store_name)
    os.makedirs(os.path.join(args.save_dir), exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(os.path.join(args.save_dir, 'log.txt'), 'w', encoding='utf-8'))
    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def setup_logging(log_file):
    """设置日志配置"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main_worker(gpu, ngpus_per_node, args):
    # 设置日志
    if args.extract_feature:
        log_file = os.path.join(args.root_log, args.store_name, "log_val.txt")
    else:
        log_file = os.path.join(args.root_log, args.store_name, "log.txt")
    logger = setup_logging(log_file)
    logger_run = None
    # logger.info(args.logger)
    if args.logger == "neptune":
        if args.ne_run != None:

            logger_run = neptune.init_run(with_id=args.ne_run, project=args.ne_project,
                                          api_token=args.ne_token,
                                          description=args.file_name)
        else:
            logger_run = neptune.init_run(project=args.ne_project,
                                          api_token=args.ne_token,
                                          description=args.file_name
                                          )

        logger_run["dataset"] = args.dataset
        logger_run["arch"] = args.arch
        logger_run["epoch"] = args.epochs
        logger_run["scaling_factor"] = args.scaling_factor
        logger_run["l_d_warm"] = args.l_d_warm
        logger_run["topk"] = args.topk
        logger_run["prob"] = args.cutmix_prob
        logger_run["args"] = args
        logger_run["tau"] = args.tau

    args.gpu = gpu
    # if args.gpu is not None:
    #     logger.info("Use GPU: {} for training".format(args.gpu))

    # create model
    # logger.info("=> creating model '{}'".format(args.arch))

    if args.arch == 'resnet50':
        model = BCLModel(name='resnet50', feat_dim=args.feat_dim,
                         num_classes=args.num_classes,
                         use_norm=args.use_norm,
                         return_features=args.extract_feature and args.extract_type == "classification" or args.knowledge_base != "")
    elif args.arch == 'resnext50':
        model = BCLModel(name='resnext50', feat_dim=args.feat_dim, num_classes=args.num_classes,
                         use_norm=args.use_norm,
                         return_features=args.extract_feature and args.extract_type == "classification" or args.knowledge_base != "")
    elif args.arch == 'resnet32':
        model = BCLModel_32(name='resnet32', feat_dim=args.feat_dim,
                            num_classes=args.num_classes,
                            use_norm=args.use_norm,
                            return_features=args.extract_feature and args.extract_type == "classification" or args.knowledge_base != "")
    elif args.arch == "resnet152":
        model = BCLModel(name='resnet152', feat_dim=args.feat_dim,
                         num_classes=args.num_classes,
                         use_norm=args.use_norm,
                         return_features=args.extract_feature and args.extract_type == "classification" or args.knowledge_base != "")
    elif args.arch == 'resnext101':
        model = BCLModel(name='resnext101', feat_dim=args.feat_dim,
                         num_classes=args.num_classes,
                         use_norm=args.use_norm,
                         return_features=args.extract_feature and args.extract_type == "classification" or args.knowledge_base != "")
    else:
        raise NotImplementedError('This model is not supported')
    # print(model)
    model_head = model.fc

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_head = model_head.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids).cuda()
        model_head = torch.nn.DataParallel(model_head, device_ids=args.device_ids).cuda()

    # model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_acc1 = 0.0
    best_scl = np.inf

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1'] if 'best_acc1' in checkpoint else 0.0
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # logger.info("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
            # logger.info("best_acc1 {}".format(best_acc1))
        else:
            pass
            # logger.info("=> no checkpoint found at '{}'".format(args.resume))
    elif args.auto_resume:
        filename = os.path.join(args.root_log, args.store_name, 'ConCutMix_ckpt.pth.tar')
        if os.path.isfile(filename):
            # logger.info("=> auto loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # logger.info("best_acc1", best_acc1)
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # logger.info("=> loaded checkpoint '{}' (epoch {})"
            #       .format(filename, checkpoint['epoch']))
        else:
            pass
            # logger.info("=> no auto checkpoint found at '{}'".format(filename))
    elif args.reload_torch:
        state_dict = model.state_dict()
        state_dict_imagenet = torch.load(args.reload_torch)
        for key in state_dict.keys():
            newkey = key[8:]
            if newkey in state_dict_imagenet.keys() and state_dict[key].shape == state_dict_imagenet[newkey].shape:
                state_dict[key] = state_dict_imagenet[newkey]
                # logger.info(newkey + " ****loaded******* ")
            else:
                pass
                # logger.info(key + " ****unloaded******* ")
        model.load_state_dict(state_dict)
    # cudnn.benchmark = True

    normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192)) if args.dataset == 'inat' \
        else transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    rgb_mean = (0.466, 0.471, 0.380) if args.dataset == 'inat' else (0.485, 0.456, 0.406)
    if args.dataset in ["Cifar100", "Cifar100-LT"]:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        rgb_mean = (0.4914, 0.4822, 0.4465)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    if not os.path.exists('{}/'.format(os.path.join(args.root_log, args.store_name))):  # 判断所在目录下是否有该文件名的文件夹
        os.makedirs(os.path.join(args.root_log, args.store_name), exist_ok=True)

    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    Uncut_augmentation_regular = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # add AutoAug
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    if args.cl_views == 'sim-sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'sim-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'randstack-randstack':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_randnclsstack), ]
    elif args.cl_views == "uncutout-sim":
        transform_train = [transforms.Compose(Uncut_augmentation_regular), transforms.Compose(augmentation_sim_cifar),
                           transforms.Compose(augmentation_sim_cifar), ]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")
    if args.dataset in ["Cifar100", "Cifar100-LT"]:
        transform_train = [transforms.Compose(Uncut_augmentation_regular), transforms.Compose(augmentation_sim_cifar),
                           transforms.Compose(augmentation_sim_cifar), ]

    txt_train, txt_val = "", ""
    if args.dataset == 'inat':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_train = f'../dataset/iNaturalist18/iNaturalist18_train.txt'
        txt_val = f'../dataset/iNaturalist18/iNaturalist18_val.txt'
        val_dataset = INaturalist(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args
        )

        train_dataset = INaturalist(
            root=args.data,
            txt=txt_train,
            args=args,
            transform=transform_train
        )
    elif args.dataset == 'iNaturalist':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_train = f'/home/Users/dqy/Projects/ConCutMix/dataset/iNaturalist18/iNaturalist18_train.txt'
        txt_val = f'/home/Users/dqy/Projects/ConCutMix/dataset/iNaturalist18/iNaturalist18_val.txt'
        val_dataset = INaturalist(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args
        )

        train_dataset = INaturalist(
            root=args.data,
            txt=txt_train,
            args=args,
            transform=transform_train
        )
    elif args.dataset == 'imagenet':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_val = f'../dataset/ImageNet_LT/ImageNet_LT_val.txt'
        txt_train = f'../dataset/ImageNet_LT/ImageNet_LT_train.txt'
        train_dataset = ImageNetLT(
            root=args.data,
            args=args,
            txt=txt_train,
            transform=transform_train)
        val_dataset = ImageNetLT(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args)
    elif args.dataset == 'ImageNet100':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_val = "/home/Users/dqy/Dataset/ImageNet100/images/val.txt"
        txt_train = "/home/Users/dqy/Dataset/ImageNet100/images/train.txt"
        train_dataset = ImageNet100(
            root=args.data,
            args=args,
            txt=txt_train,
            transform=transform_train)
        val_dataset = ImageNet100(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args)
    elif args.dataset == 'ImageNet100-LT':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_val = "/home/Users/dqy/Dataset/ImageNet100-LT/format_ImageNet/images/val.txt"
        txt_train = "/home/Users/dqy/Dataset/ImageNet100-LT/format_ImageNet/images/train.txt"
        train_dataset = ImageNet100(
            root=args.data,
            args=args,
            txt=txt_train,
            transform=transform_train)
        val_dataset = ImageNet100(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args)

    elif args.dataset == 'cifar10':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_dataset = IMBALANCECIFAR10(root=args.data, args=args,
                                       transform=val_transform,
                                       train=False, imb_factor=1, download=True)
        train_dataset = IMBALANCECIFAR10(
            root=args.data, args=args, download=True,
            imb_factor=args.imb_factor,
            transform=transform_train)
    elif args.dataset == 'cifar100':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_dataset = IMBALANCECIFAR100(root=args.data, args=args,
                                        download=True,
                                        transform=val_transform,
                                        train=False, imb_factor=1)
        train_dataset = IMBALANCECIFAR100(
            root=args.data, args=args,
            download=True,
            imb_factor=args.imb_factor,
            transform=transform_train)
    elif args.dataset == 'Cifar100':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        txt_val = "/home/Users/dqy/Dataset/Cifar/format_ImageNet/images/val.txt"
        txt_train = "/home/Users/dqy/Dataset/Cifar/format_ImageNet/images/train.txt"
        train_dataset = Cifar100(
            root=args.data,
            args=args,
            txt=txt_train,
            transform=transform_train)
        val_dataset = Cifar100(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args)
    elif args.dataset == 'Cifar100-LT':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        txt_val = "/home/Users/dqy/Dataset/Cifar100-LT/format_ImageNet/images/val.txt"
        txt_train = "/home/Users/dqy/Dataset/Cifar100-LT/format_ImageNet/images/train.txt"
        train_dataset = Cifar100(
            root=args.data,
            args=args,
            txt=txt_train,
            transform=transform_train)
        val_dataset = Cifar100(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args)
    elif args.dataset == 'Places_LT' or args.dataset == 'Places365-LT':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_val = "/home/Users/dqy/Dataset/Places365-LT/places365_LT/val.txt"
        txt_train = "/home/Users/dqy/Dataset/Places365-LT/places365_LT/train.txt"
        train_dataset = PlacesLT(
            root=args.data,
            args=args,
            txt=txt_train,
            transform=transform_train)
        val_dataset = PlacesLT(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args)
    elif args.dataset == 'Places365':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_val = "/home/Users/dqy/Dataset/Places365/format_ImageNet/images/val.txt"
        txt_train = "/home/Users/dqy/Dataset/Places365/format_ImageNet/images/train.txt"
        train_dataset = PlacesLT(
            root=args.data,
            args=args,
            txt=txt_train,
            transform=transform_train)
        val_dataset = PlacesLT(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False, args=args)
    else:
        raise ValueError(f"Not implemented dataset {args.dataset}")

    cls_num_list = train_dataset.cls_num_list
    args.cls_num = len(cls_num_list)
    # logger.info(len(cls_num_list))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    criterion_scl = BalSCL(cls_num_list, args.temp).cuda(args.gpu)
    criterion_ce = LogitAdjust(cls_num_list, tau=args.tau).cuda(args.gpu)
    criterion_ce_cutmix = cutmix_cross_entropy(cls_num_list, args.tau).cuda(args.gpu)

    if args.reload:
        test_dataset = None
        if args.dataset == 'inat':
            txt_test = f'../dataset/iNaturalist18/iNaturalist18_val.txt'
            test_dataset = INaturalist(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args)
        elif (args.dataset == 'iNaturalist'):
            txt_test = txt_val if args.extract_phase == "val" else txt_train
            test_dataset = INaturalist(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args
            )
        elif args.dataset == 'ImageNet100':
            txt_test = txt_val if args.extract_phase == "val" else txt_train
            test_dataset = ImageNet100(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args)
        elif args.dataset == 'ImageNet100-LT':
            txt_test = txt_val if args.extract_phase == "val" else txt_train
            test_dataset = ImageNet100(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args)
        elif args.dataset == 'Cifar100':
            txt_test = txt_val if args.extract_phase == "val" else txt_train
            test_dataset = Cifar100(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args)
        elif args.dataset == 'Cifar100-LT':
            txt_test = txt_val if args.extract_phase == "val" else txt_train
            test_dataset = Cifar100(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args)
        elif args.dataset == 'imagenet':
            txt_test = f'../dataset/ImageNet_LT/ImageNet_LT_test.txt'
            test_dataset = ImageNetLT(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args)
        elif args.dataset == 'cifar10':
            test_dataset = IMBALANCECIFAR10(root=args.data, args=args, transform=val_transform,
                                            train=args.extract_phase == "train",
                                            imb_factor=1 if args.extract_phase == "val" else 0.1, download=True)
        elif args.dataset == 'cifar100':
            test_dataset = IMBALANCECIFAR100(root=args.data, args=args,
                                            download=True,
                                            transform=val_transform,
                                            train=args.extract_phase == "train", imb_factor=1)
        elif args.dataset == 'Places_LT' or args.dataset == 'Places365-LT':
            txt_test = txt_val if args.extract_phase == "val" else txt_train
            test_dataset = PlacesLT(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args)
        elif args.dataset == 'Places365':
            txt_test = txt_val if args.extract_phase == "val" else txt_train
            test_dataset = PlacesLT(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False, args=args)
        else:
            raise ValueError(f"Not implemented dataset {args.dataset}")

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        acc1, many, med, few, class_acc, scl_loss = validate(train_loader, test_loader, model, criterion_ce, criterion_scl, 1, args, logger=logger, model_head=model_head)
        # logger.info('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}, SCL loss: {:.3f}'
        #       .format(acc1, many, med, few, scl_loss))
        return
    logger.info("start train")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)
        ce_loss_all, scl_loss_all, top1, loss = train(train_loader, model, criterion_ce, criterion_ce_cutmix,
                                                      criterion_scl, optimizer,
                                                      epoch, args,
                                                      logger_run, cls_num_list, logger)
        # evaluate on validation set
        acc1, many, med, few, class_acc, scl_loss = validate(train_loader, val_loader, model, criterion_ce, criterion_scl, epoch, args, logger=logger
                                                   )
        if (args.logger == "neptune"):
            logger_run["few_acc"].log(few, step=epoch)
            logger_run["val_acc"].log(acc1, step=epoch)
            logger_run["many_acc"].log(many, step=epoch)
            logger_run["median_acc"].log(med, step=epoch)
            logger_run["CE_loss/train"].log(ce_loss_all, step=epoch, )
            logger_run["SCL_loss/train"].log(scl_loss_all, step=epoch)
            logger_run["train_acc"].log(top1, step=epoch)
            logger_run["train_loss"].log(loss, step=epoch)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
            best_class_acc = class_acc
            if (logger_run != None):
                logger_run["few_acc_top1"].log(best_few, step=epoch)
                logger_run["val_acc_top1"].log(best_acc1, step=epoch)
                logger_run["many_acc_top1"].log(best_many, step=epoch)
                logger_run["median_acc_top1"].log(best_med, step=epoch)
            logger.info(
                'Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(
                    best_acc1,
                    best_many,
                    best_med,
                    best_few,
                ))
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        # remember best scl and save checkpoint
        is_best_scl = scl_loss < best_scl
        best_scl = min(scl_loss, best_scl)
        if is_best_scl:
            if (logger_run != None):
                logger_run["val/best_scl_loss"].log(best_scl, step=epoch)
            logger.info(
                'Best SCL loss: {:.3f}'.format(
                    best_scl
                ))
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_scl_loss': best_scl,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion_ce, criterion_ce_cutmix, criterion_scl, optimizer, epoch,
          args,
          logger_run, cls_num_list, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    end = time.time()

    model.train()
    for i, data in enumerate(train_loader):
        sample_A, sample_B, target_A, target_B = data  # !Modified: n_views args testing
        batch_size = target_A.shape[0]
        target_A, target_B = target_A.cuda(), target_B.cuda()
        # cutmix
        sample_A[0], sample_A[1], sample_A[2], = sample_A[0].cuda(), sample_A[1].cuda(), sample_A[2].cuda()
        sample_B[0], sample_B[1], sample_B[2], = sample_B[0].cuda(), sample_B[1].cuda(), sample_B[2].cuda()
        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(sample_B[0].size()[0]).cuda()
        r = np.random.rand(1)

        if r < args.cutmix_prob:
            target_a = target_A
            target_b = target_B[rand_index]
            ta = torch.nn.functional.one_hot(target_a, num_classes=args.num_classes)
            tb = torch.nn.functional.one_hot(target_b, num_classes=args.num_classes)
            bbx1, bby1, bbx2, bby2 = rand_bbox(sample_B[0].size(), lam)
            cutmix_sample1 = sample_A[0].clone()
            cutmix_sample1[:, :, bbx1:bbx2, bby1:bby2] = sample_B[0][rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (sample_B[0].size()[-1] * sample_B[0].size()[-2]))
            target_cutmix = (torch.tensor([lam]).cuda() * ta) + ((1 - torch.tensor([lam])).cuda() * tb)
            inputs = torch.cat([cutmix_sample1, sample_A[1], sample_A[2]], dim=0)
            inputs = inputs.cuda()
            feat_mlp, logits, centers, uncenters, unfeat = model(inputs)
            uncenters = uncenters[:args.cls_num]
            logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            f1, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            unfeat1, unfeat2, unfeat3 = torch.split(unfeat, [batch_size, batch_size, batch_size], dim=0)

            if ((epoch > args.l_d_warm)):

                target_lam = get_semantically_consistent_label(unfeat1, uncenters, target_cutmix, args.scaling_factor,
                                                               cls_num_list, args.topk)
                ce_loss = criterion_ce_cutmix(logits, target_lam)
            else:
                ce_loss = criterion_ce(logits, target_a) * lam + criterion_ce(logits, target_b,
                                                                              ) * (1. - lam)

            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            centers = centers[:args.cls_num]
            scl_loss = criterion_scl(centers, features, target_A, )
        else:
            inputs = torch.cat([sample_A[0], sample_A[1], sample_A[2]], dim=0)
            inputs = inputs.cuda()
            feat_mlp, logits, centers, uncenter, unfeat = model(inputs)
            logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            centers = centers[:args.cls_num]
            uncenter = uncenter[:args.cls_num]
            ce_loss = criterion_ce(logits, target_A)
            # print(centers.shape, features.shape, target_A.shape)
            scl_loss = criterion_scl(centers, features, target_A, )

        loss = args.alpha * ce_loss + args.beta * scl_loss

        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)
        # 梯度累积
        if (args.grad_c):
            loss = loss / (256 / batch_size)
            loss.backward()
            if ((i + 1) % (256 / batch_size) == 0):
                optimizer.step()
                optimizer.zero_grad()


        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'loss {loss:.4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, loss=loss))
            logger.info(output)

        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)

        acc1 = accuracy(logits, target_A, topk=(1,))
        top1.update(acc1[0].item(), batch_size)
    output = ('Epoch Summary: [{0}][{1}/{2}] \t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
              'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'loss {loss:.4f}'.format(
        epoch, i, len(train_loader), batch_time=batch_time,
        ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, loss=loss))
    logger.info(output)
    return ce_loss_all.avg, scl_loss_all.avg, top1.avg, loss


def validate(train_loader, val_loader, model, criterion_ce, criterion_scl, epoch, args, flag='val', logger=None,
             model_head=None):
    model.eval()
    all_preds, all_labels = [], []
    results = []
    total_loss = 0.0
    num_samples = 0
    total_code_words = 0
    retrieved_code_words = set()
    retrieval_count_per_word = {}

    if args.knowledge_base:
        # CLIP_model, CLIP_preprocess = load_CLIP_model(device)
        Contrast_model = load_Contrast_model()
        Contrast_model = torch.nn.DataParallel(Contrast_model, device_ids=args.device_ids).cuda()
        Contrast_model.eval()
        faiss_folder = os.path.join(args.knowledge_base, "Faiss_index")
        Contrast_paths = np.load(os.path.join(faiss_folder, "index_paths.npy"), allow_pickle=True)
        total_code_words = len(Contrast_paths)
        print(f"[INFO] 知识库总码字数: {total_code_words}")

    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    # for feature extraction
    if args.extract_feature:
        os.makedirs(args.save_dir, exist_ok=True)
    label2category = label_to_name(args)
    label_map = name_to_label()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(tqdm.tqdm(val_loader)):
            inputs, targets, categories, names = data
            inputs, targets = inputs.cuda(), targets.cuda()
            images = inputs
            img_names = names
            labels = targets
            batch_size = targets.size(0)
            if not args.knowledge_base:
                if args.extract_feature and args.extract_type =="classification":
                    feat_mlp, logits, centers, _, _, features_encoder = model(inputs)
                else:
                    feat_mlp, logits, centers, _, _ = model(inputs)
                outputs = logits

                ce_loss = criterion_ce(logits, targets)
                centers = centers[:args.cls_num]
                features = feat_mlp[:, None, :].repeat((1, 2, 1))
                target_A = targets
                scl_loss = criterion_scl(centers, features, target_A, )
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, targets))

                acc1 = accuracy(logits, targets, topk=(1,))
                ce_loss_all.update(ce_loss.item(), batch_size)
                scl_loss_all.update(scl_loss.item(), batch_size)
                top1.update(acc1[0].item(), batch_size)

                batch_time.update(time.time() - end)

                # if i % args.print_freq == 0:
                #     output = ('Test: [{0}/{1}]\t'
                #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #               'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                #               'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #     .format(
                #         i, len(val_loader), batch_time=batch_time, ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1,
                #     ))
                #     logger.info(output)
                if args.extract_feature and args.extract_type == "contrast":  # 提取对比特征
                    for b in range(inputs.size(0)):
                        save_path = os.path.join(args.save_dir, categories[b], f"{names[b]}.npy")
                        os.makedirs(os.path.join(args.save_dir, categories[b]), exist_ok=True)
                        np.save(save_path, feat_mlp[b].cpu().numpy())
                if args.extract_feature and args.extract_type =="classification":  # 提取分类特征（经分类头前）
                    feats = features_encoder.cpu().numpy()
                    # 获取预测结果
                    probs = F.softmax(logits, dim=1)
                    pred_probs, pred_indices = torch.max(probs, dim=1)
                    for feat, name, label, output, pred_prob, pred_idx in zip(
                        feats, names, targets, logits, pred_probs, pred_indices):
                        
                        # 转换为 numpy 和 Python 标量
                        feat_np = feat
                        label_np = int(label.cpu().numpy())
                        output_np = output.cpu().numpy()
                        pred_idx_np = int(pred_idx.cpu().numpy())
                        pred_prob_np = float(pred_prob.cpu().numpy())
                        
                        # 获取类别信息
                        actual_category = str(label_np)
                        actual_label_id = int(label_np)
                        pred_category = str(pred_idx_np)
                        pred_label_id = int(pred_idx_np)
                        
                        # 保存特征
                        os.makedirs(os.path.join(args.save_dir, "ConCutMix_features", actual_category), exist_ok=True)
                        np.save(os.path.join(args.save_dir, "ConCutMix_features", actual_category, f"{name.split('.')[0]}.npy"), feat_np)
                        
                        # 保存 logits 和预测信息
                        os.makedirs(os.path.join(args.save_dir, "Logits", actual_category), exist_ok=True)
                        
                        # 创建保存字典
                        save_data = {
                            'logits': output_np,  # logits 输出
                            'predicted_category': pred_category,  # 预测类别名称
                            'predicted_label_id': pred_label_id,  # 预测类别ID
                            'predicted_probability': pred_prob_np,  # 预测概率
                            'actual_category': actual_category,  # 实际类别名称
                            'actual_label_id': actual_label_id,  # 实际类别ID
                            'image_name': name,  # 图像文件名
                            'feature_shape': feat_np.shape  # 特征维度
                        }
                        
                        # 保存为 npz 文件（可以保存多个数组）
                        np.savez(
                            os.path.join(args.save_dir, "Logits", actual_category, f"{name.split('.')[0]}.npz"),
                            **save_data
                        )
            else:
                # 建立 Faiss 索引库
                faiss_folder = os.path.join(args.knowledge_base, "Faiss_index")
                assert os.path.exists(os.path.join(faiss_folder, "Contrast_index.faiss"))
                assert os.path.exists(os.path.join(faiss_folder, "index_paths.npy"))
                assert os.path.exists(os.path.join(args.knowledge_base, "Contrast_features"))
                assert os.path.exists(os.path.join(args.knowledge_base, "ConCutMix_features"))
                Contrast_paths = np.load(os.path.join(faiss_folder, "index_paths.npy"), allow_pickle=True)
                # 提取图像 Contrast 特征以供索引
                # Contrast_features = Contrast_model(images).squeeze().cpu().numpy().astype('float32')
                Contrast_features, _, _, _, _ = Contrast_model(renormalize_images_simple(images))
                Contrast_features = Contrast_features.squeeze().cpu().numpy().astype('float32')
                # 通过 Faiss 向量库进行索引
                D, I = retrieval_Faiss(faiss_folder, Contrast_features, k=args.retrieval_k)
                # 将当前batch中检索到的码字索引添加到集合中
                batch_retrieved_indices = set(I.flatten())
                retrieved_code_words.update(batch_retrieved_indices)
                
                # 统计每个码字被检索的次数
                for indices in I:
                    for idx in indices:
                        retrieval_count_per_word[idx] = retrieval_count_per_word.get(idx, 0) + 1
                # 仅基于检索结果进行分类
                if args.only_retrieval:
                    # 从检索结果中推断类别
                    for i, (dists, indices) in enumerate(zip(D, I)):
                        label_count = {}
                        label_dists = {}
                        for dist, idx in zip(dists, indices):
                            path = Contrast_paths[idx]  # e.g., "kernal_0#n01775062#n01775062_4379.pt"
                            filename = os.path.basename(path)
                            category = filename.split("#")[1]
                            label = label_map[category]

                            label_count[label] = label_count.get(label, 0) + 1
                            label_dists.setdefault(label, []).append(dist)

                        # 找到出现频率最高的类别标签
                        max_count = max(label_count.values())
                        candidates = [lbl for lbl, count in label_count.items() if count == max_count]

                        if len(candidates) == 1:
                            final_label = candidates[0]
                        else:
                            # 多个频率最高的候选标签，选平均相似度最高的
                            avg_similarities = {
                                lbl: np.mean(label_dists[lbl]) for lbl in candidates
                            }
                            final_label = max(avg_similarities.items(), key=lambda x: x[1])[0]

                        all_preds.append(torch.tensor([final_label]))
                        results.append({
                            "image": img_names[i],
                            "ground_truth": int(labels[i].cpu().item()),
                            "predicted": int(final_label),
                            "retrieved_indices": [int(idx) for idx in indices],
                            "similarities": [float(sim) for sim in dists]
                        })
                    all_labels.append(labels.cpu())
                    num_samples += images.size(0)
                    continue  # 跳过后续标准模型推理流程
                # 否则，根据索引结果获取 ResNet 特征并进行加权
                assert model_head is not None
                feat_mlp, logits, centers, _, _, features_encoder = model(inputs)
                feats = features_encoder
                features_prior = []
                for similarities, indexes in zip(D, I):  # 处理 batch 中各个样本的检索结果
                    weights = softmax(similarities)
                    ConCutMix_features = []
                    for index in indexes:
                        Contrast_path = Contrast_paths[index]
                        feature_path = Contrast_path.replace("Contrast_features", "ConCutMix_features")
                        ConCutMix_features.append(np.load(feature_path))
                    weighted_prior = np.sum(weights[:, None] * np.stack(ConCutMix_features), axis=0)
                    features_prior.append(weighted_prior)
                # 根据加权结果进一步调整图像特征
                feats = feats * (1 - args.prior_weight) + torch.Tensor(np.array(features_prior)).to(feats.device) * args.prior_weight
                outputs = model_head(feats[:, :])
            loss = criterion_ce(outputs, labels)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            for i in range(len(img_names)):
                record = {
                    "category": label2category[labels[i].cpu().item()],
                    "image": img_names[i],
                    "ground_truth": int(labels[i].cpu().item()),
                    "predicted": int(preds[i].cpu().item()),
                }
                if args.knowledge_base:
                    record.update({
                        "retrieved_indices": [int(idx) for idx in I[i]],
                        "similarities": [float(sim) for sim in D[i]]
                        })
                results.append(record)

        # probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        # many_acc_top1, median_acc_top1, low_acc_top1, class_acc = shot_acc(preds, total_labels, train_loader,
        #                                                                    acc_per_cls=False)

        # output = ('Test Summary: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
        #           'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        # .format(
        #     i, len(val_loader), batch_time=batch_time, ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1,
        # ))
        # logger.info(output)
    avg_loss = total_loss / num_samples
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, labels=sorted(label_map.keys()), digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_map.values()))
    print(f"[RESULT] Evaluating...")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_p:.4f} | Recall: {macro_r:.4f} | F1: {macro_f1:.4f}")
    print(f"Weighted Precision: {w_p:.4f} | Recall: {w_r:.4f} | F1: {w_f1:.4f}")
    # print(report)
    if args.result_json_path:
        import json
        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Prediction results saved to: {args.result_json_path}")
    # 码字利用率相关输出
    if args.knowledge_base and total_code_words > 0:
        # 计算利用率指标
        utilized_count = len(retrieved_code_words)
        utilization_rate = utilized_count / total_code_words * 100
        
        # 计算检索频率统计
        retrieval_counts = list(retrieval_count_per_word.values())
        if retrieval_counts:
            avg_retrieval_per_word = int(np.mean(retrieval_counts))
            max_retrieval_per_word = int(np.max(retrieval_counts))
            min_retrieval_per_word = int(np.min(retrieval_counts))
            
            # 计算检索分布
            retrieval_distribution = {}
            for count in retrieval_counts:
                retrieval_distribution[count] = retrieval_distribution.get(count, 0) + 1
        else:
            avg_retrieval_per_word = max_retrieval_per_word = min_retrieval_per_word = 0
            retrieval_distribution = {}
        
        # print("\n" + "="*60)
        # print("知识库码字利用率统计")
        # print("="*60)
        # print(f"知识库总码字数: {total_code_words}")
        # print(f"被检索到的码字数: {utilized_count}")
        # print(f"码字利用率: {utilization_rate:.2f}%")
        # print(f"平均每个码字被检索次数: {avg_retrieval_per_word:.2f}")
        # print(f"单个码字最大检索次数: {max_retrieval_per_word}")
        # print(f"单个码字最小检索次数: {min_retrieval_per_word}")
        
        # # 输出检索分布
        # print(f"\n检索次数分布:")
        # for count in sorted(retrieval_distribution.keys()):
        #     percentage = retrieval_distribution[count] / total_code_words * 100
        #     print(f"  检索{count}次: {retrieval_distribution[count]}个码字 ({percentage:.1f}%)")
        
        # # 输出最常被检索的码字（前10个）
        # if retrieval_count_per_word:
        #     print(f"\n最常被检索的码字 (前10):")
        #     sorted_retrieval = sorted(retrieval_count_per_word.items(), key=lambda x: x[1], reverse=True)[:10]
        #     for idx, count in sorted_retrieval:
        #         path = Contrast_paths[idx] if idx < len(Contrast_paths) else "Unknown"
        #         print(f"  码字{idx}: {path} - 被检索{count}次")
        
        # 将利用率统计保存到结果文件中
        if args.result_json_path:
            utilization_stats = {
                "total_code_words": total_code_words,
                "utilized_code_words": utilized_count,
                "utilization_rate": utilization_rate,
                "avg_retrieval_per_word": avg_retrieval_per_word,
                "max_retrieval_per_word": max_retrieval_per_word,
                "min_retrieval_per_word": min_retrieval_per_word,
                "retrieval_distribution": retrieval_distribution
            }
            
            # 如果结果文件已存在，则读取并更新统计信息
            if os.path.exists(args.result_json_path):
                with open(args.result_json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                existing_data = {"retrieval_info": existing_data}
                existing_data["utilization_stats"] = utilization_stats
                with open(args.result_json_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2)
            else:
                # 创建新的结果文件包含统计信息
                result_data = {
                    "validation_results": results,
                    "utilization_stats": utilization_stats
                }
                with open(args.result_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2)
            
            print(f"[INFO] 码字利用率统计已保存到: {args.result_json_path}")
    return acc, 0, 0, 0, 0, 0


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.root_log, args.store_name, 'ConCutMix_ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        if 'best_scl_loss' in state:
            shutil.copyfile(filename, filename.replace('pth.tar', 'best_scl.pth.tar'))
        else:
            shutil.copyfile(filename, filename.replace('pth.tar', 'best_acc1.pth.tar'))


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
            # lr *= 0.1 if epoch == milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_semantically_consistent_label(feature, center, target, scaling_factor, cls, k):
    #get the scaling factor omega
    scaling_factor = scaling_factor[0] / scaling_factor[1]
    #get N
    cls_num_list = torch.cuda.FloatTensor(cls)
    #sum(log(N_i))
    weight = torch.log((cls_num_list * target).sum(1))
    #N /sum(log(N_i))
    weight = (weight / (torch.log(cls_num_list).sum())).reshape(-1, 1)
    target_de = target.detach()
    center_de = center.detach()
    feature_de = feature.detach()
    # get the euclidean distance
    sim = torch.sqrt(torch.sum((feature_de[:, None, :] - center_de) ** 2, dim=2))
    sim = 1 / sim
    # top K
    indices_to_remove = sim < torch.topk(sim, k)[0][..., -1, None]
    sim[indices_to_remove] = 0
    final_sim = sim
    # normlaization
    label = F.normalize(final_sim, p=1, dim=1)
    label = (weight * scaling_factor) * label + (1 - weight * scaling_factor) * target_de
    return label


if __name__ == '__main__':
    main()
    # center=torch.tensor([[1.21,1],[1,4],[2,5],[3,5]])
    # target=torch.tensor([0,2,0,1,2])
    # feature=torch.tensor([[1.1,1],[2,6],[2,50],[33,5],[34,1]])
    # target=F.one_hot(target,4)
    # print(get_semantically_consistent_label(feature,center,target,0.01,[100,10,5,1],2))
    # targetA = torch.tensor([0, 2, 0, 3])
    # targetB = torch.tensor([1, 0, 2, 1])
    # lam=0.1
    # center = torch.tensor([[1.21, 1], [1, 4], [2, 5], [3, 5]])
    # print(get_distance_cutmix_5(center, targetA,targetB,lam, 0.01, [9, 6, 3, 2]))
    exit(0)
