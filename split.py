import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models
from models.resnet_simclr import ResNetBertSimCLR, ResNetSimCLR
from simclr import BertSimCLR, SimCLR


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')

def main():
    args = parser.parse_args()
    print("Creating model")
    model = ResNetBertSimCLR(base_model=args.arch, out_dim=args.out_dim)
    print("Model created")
    checkpoint = torch.load('/home/pliang/gabriel/BertSimCLR/runs/Sep09_19-15-40_quad-p40-0-1/checkpoint_0001.pth.tar')#, map_location="cuda:0")
    print("Model loaded")
    model_state = checkpoint['state_dict']#.to(args.device)
    model.load_state_dict(model_state)
    print(model)
    #model.fc = nn.Linear(2048, args.out_dim)
    #torch.save(model.state_dict(), 'split_model.pth')


if __name__ == "__main__":
    main()
