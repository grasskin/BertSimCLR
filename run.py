import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, coco_collate_fn
from models.resnet_simclr import ResNetBertSimCLR, ResNetSimCLR
import faulthandler; faulthandler.enable()
from simclr import BertSimCLR, SimCLR
from torchvision.datasets import CIFAR10

from torchvision import transforms
import torch.nn as nn

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-d', '--dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10', 'mscoco', 'mscocobaseline'])
parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=210, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('-C', default=1, type=int, help='Amount of multimodal loss.')
parser.add_argument('--eval', default=False, type=bool, help='Run linear classifier evaluation.')
parser.add_argument('--saved-path', default='', help='Path to saved checkpoint.')
parser.add_argument('--baseline', default=False, type=bool, help='Run baseline SimCLR training')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
        torch.multiprocessing.set_start_method('spawn')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    if args.eval:
        # Load pretrained model and cifar10

        cifar_transforms = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])
        dataset = CIFAR10(root='datasets/cifar10', download=True, transform=cifar_transforms)
        train_dataset = CIFAR10(root='datasets/cifar10', train=True, transform=cifar_transforms)
        valid_dataset = CIFAR10(root='datasets/cifar10', train=False, transform=cifar_transforms)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)
        if not args.baseline:
            model = ResNetBertSimCLR(base_model=args.arch, out_dim=args.out_dim)
            checkpoint = torch.load(args.saved_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
            checkpoint = torch.load(args.saved_path)
            model.load_state_dict(checkpoint['state_dict'])           

        classifier_model = torch.nn.Sequential(torch.nn.Linear(128, 10))
        classifier_optimizer = torch.optim.Adam(classifier_model.parameters(), args.lr, weight_decay=args.weight_decay)

        optimizer = None
        scheduler = None
    else:
        # Load BertSimCLR and coco dataset
        dataset = ContrastiveLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
        valid_dataset = dataset.get_dataset(args.dataset_name+'valid', args.n_views)

        if not args.baseline:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=coco_collate_fn)
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=coco_collate_fn)
            model = ResNetBertSimCLR(base_model=args.arch, out_dim=args.out_dim)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True)

            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True)
            model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

        
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

        # emergency fix

        model.to(args.device)
        checkpoint = torch.load('/home/pliang/gabriel/BertSimCLR/runs/Sep09_19-15-40_quad-p40-0-1/checkpoint_0048.pth.tar')#, map_location="cuda:0")
        model_state = checkpoint['state_dict']#.to(args.device)
        opt_state = checkpoint['optimizer']#.to(args.device)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(opt_state)
        model.to(args.device)


        classifier_model = None
        classifier_optimizer = None

    #print(model.visual_backbone)

    data_loaders = {"train": train_loader, "val": valid_loader}

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        if not args.baseline:
            simclr = BertSimCLR(model=model, optimizer=optimizer, scheduler=scheduler, classifier_model=classifier_model, classifier_optimizer=classifier_optimizer, args=args)
            if args.eval:
                simclr.train_linear_classifier(args.epochs, data_loaders)
            else:
                simclr.train(data_loaders)
        else:
            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, classifier_model=classifier_model, classifier_optimizer=classifier_optimizer, args=args)
            if args.eval:
                simclr.train_linear_classifier(args.epochs, data_loaders)
            else:
                simclr.train(data_loaders)

if __name__ == "__main__":
    main()
