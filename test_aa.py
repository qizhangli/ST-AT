import torch
import torchvision

from models.resnet import ResNet18
from models.wideresnet import WideResNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', default=None, type=str, )
parser.add_argument('--log', default=None, type=str, )
parser.add_argument('--eps', default=8/255, type=float, )
parser.add_argument('--dataset', default="cifar10", type=str, )
parser.add_argument('--arch', default="resnet18", type=str, )
args = parser.parse_args()

print("epsilon:", args.eps)

if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor())
elif args.dataset == "cifar100":
    testset = torchvision.datasets.CIFAR100("data", train=False, transform=torchvision.transforms.ToTensor())
elif args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root='data/svhn', split="test", download=False, transform=torchvision.transforms.ToTensor())


testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=len(testset),
                                            num_workers=8, pin_memory=True)

num_classes = 10 if args.dataset != "cifar100" else 100
if args.arch == "resnet18":
    model = ResNet18(num_classes=num_classes)
    model.cuda()
    model = torch.nn.DataParallel(model)
    state_dict = torch.load(args.model_path)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
elif args.arch == "wrn-28-10":
    state_dict = torch.load(args.model_path, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
    if "module.sub_block1.layer.0.bn1.weight" in list(state_dict.keys()):
        subblock1=True
    else:
        subblock1=False
    model = WideResNet(depth=28, widen_factor=10, subblock1=subblock1)
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()

from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=args.eps, log_path=args.log, version='standard')
for i, (x_test, y_test) in enumerate(testloader):
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=2000)