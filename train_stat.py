import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.backends import cudnn
from torchvision import transforms

from losses import stat_loss
from models.resnet import *
from utils import adjust_learning_rate, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default="results/debug",
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--eval_attack_batches', default=10, type=int,
                    help='Number of eval batches to attack with PGD or certify '
                         'with randomized smoothing')
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--trades_set', default=False, action='store_true')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default="cifar10", type=str, )
parser.add_argument('--beta', default=None, type=float)

parser.add_argument('--loss_mode', default=None, type=str)
parser.add_argument('--ignore_col_bn', default=False, action='store_true')
args = parser.parse_args()

assert args.dataset in ["cifar10", "cifar100", "svhn"]


cudnn.benchmark = False
cudnn.deterministic = True
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


args.epsilon = 8./255
args.step_size = 2./255
args.test_epsilon = 8./255
args.test_step_size = 2./255
args.test_num_steps= 20
args.gaussian_init = True
args.weight_decay = 5e-4

if args.epochs == 120:
    args.lr_mode = "120e"
elif args.epochs == 80:
    args.lr_mode = "80e"

if args.dataset == "svhn":
    args.step_size = 1./255
    args.lr=0.01

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(model_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
logging.info("Input args: %r", args)

torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True}


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    eval_trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=False, transform=transform_test)
    eval_train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    eval_trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=False, transform=transform_test)
    eval_train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    testset = torchvision.datasets.CIFAR100(root='data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='data/svhn', split="train", download=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    eval_trainset = torchvision.datasets.SVHN(root='data/svhn', split="train", download=False, transform=transform_test)
    eval_train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    testset = torchvision.datasets.SVHN(root='data/svhn', split="test", download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # calculate robust loss
        (loss, 
         ce_upper, ce_clean, ce_lower) = stat_loss(model=model,
                                                    x_natural=data,
                                                    y=target,
                                                    optimizer=optimizer,
                                                    step_size=args.step_size,
                                                    epsilon=args.epsilon,
                                                    perturb_steps=args.num_steps,
                                                    beta=args.beta,
                                                    loss_mode=args.loss_mode)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCE_UPPER: {:.4f}\tCE_CLEAN: {:.4f}\tCE_LOWER: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 
                            ce_upper, ce_clean, ce_lower))



def main():
    model = ResNet18(num_classes=10 if args.dataset != "cifar100" else 100).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        state_dicts = torch.load(args.resume)
        model.load_state_dict(state_dicts["state_dict"])
        optimizer.load_state_dict(state_dicts["opt_state_dict"])
        start_epoch = 1 + state_dicts["epoch"]
    else:
        start_epoch = 1


    best_robust_acc=0
    for epoch in range(start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(args.lr, optimizer, epoch, mode=args.lr_mode)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        logging.info('================================================================')
        evaluate(args, model, device, 'train', eval_train_loader, eval_attack_batches = args.eval_attack_batches)
        clean_accuracy, robust_accuracy = evaluate(args, model, device, 'test', test_loader, eval_attack_batches = args.eval_attack_batches)
        logging.info('================================================================')

        
        # save checkpoint
        save_dict = {"state_dict": model.state_dict(),
                     "opt_state_dict": optimizer.state_dict(),
                     "epoch": epoch,
                     "robust_acc": robust_accuracy,
                     "clean_acc": clean_accuracy}
        torch.save(save_dict, 
                   os.path.join(model_dir, 'ep_cur.pt'))
        if robust_accuracy >= best_robust_acc:
            torch.save(save_dict, 
                       os.path.join(model_dir, 'ep_best.pt'))
            best_robust_acc = robust_accuracy
        if epoch % args.save_freq == 0:
            torch.save(save_dict, 
                       os.path.join(model_dir, 'ep_{}.pt'.format(epoch)))

if __name__ == '__main__':
    main()
