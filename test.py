import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from models.resnet import *
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--dataset', default="cifar10", type=str, )
parser.add_argument('--log', default=None, type=str, )
args = parser.parse_args()

args.log = args.log + "/FGSM_PGD_CW_" + args.model_path.split("/")[-2]+"_"+args.model_path.split("/")[-1].split(".")[0]+".log"

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(args.log),
            logging.StreamHandler(),
        ],
    )

# settings
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor())
elif args.dataset == "cifar100":
    testset = torchvision.datasets.CIFAR100("data", train=False, transform=torchvision.transforms.ToTensor())
elif args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root='data/svhn', split="test", download=False, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

@torch.no_grad()
def get_rank2_label(logit, y):
    batch_size = len(logit)
    tmp = logit.clone()
    # tmp = logit - logit[torch.arange(batch_size), y][:, None]
    tmp[torch.arange(batch_size), y] = -float("inf")
    return tmp.argmax(1)

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  mode):
    batch_size = len(X)
    with torch.no_grad():
        out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    if mode != "FGSM":
        random_noise = X.new(X.size()).uniform_(-epsilon, epsilon)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        adv_logit = model(X_pgd)
        if mode=="CW":
            rank2_label = get_rank2_label(adv_logit, y)
            loss = - adv_logit[torch.arange(batch_size), y] + adv_logit[torch.arange(batch_size), rank2_label]
            loss = loss.sum() / batch_size
        elif mode in ["PGD", "FGSM"]:
            loss = F.cross_entropy(adv_logit, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    logging.info('err pgd (white-box): {}, {:0.2f}'.format(err_pgd, 100 - (100*err_pgd / len(X))))
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader, epsilon, step_size, num_steps, mode):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    num_steps = int(num_steps)
    logging.info("mode: {}, epsilon: {:.6f}, step_size: {:.6f}, num_steps: {}".format(mode, epsilon, step_size, num_steps))

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, mode)
        robust_err_total += err_robust
        natural_err_total += err_natural
    logging.info('natural_err_total: {}'.format(natural_err_total))
    logging.info('robust_err_total: {}, {:0.2f}'.format(robust_err_total, 100 - (100*robust_err_total / 10000)))


def main():

    # white-box attack
    logging.info('pgd white-box attack')
    model = ResNet18(num_classes=10 if args.dataset != "cifar100" else 100).cuda()
    ckpt_dict = torch.load(args.model_path)
    state_dict = ckpt_dict["state_dict"]
    logging.info("path: {}".format(args.model_path))
    logging.info("epoch: {}".format(ckpt_dict["epoch"]))
    if "module" in list(state_dict.keys())[0]:
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
        model = nn.DataParallel(model)
    eval_adv_test_whitebox(model, device, test_loader, epsilon=8./255, step_size=8./255, num_steps=1, mode="FGSM")
    eval_adv_test_whitebox(model, device, test_loader, epsilon=8./255, step_size=2./255, num_steps=100, mode="PGD")
    eval_adv_test_whitebox(model, device, test_loader, epsilon=8./255, step_size=2./255, num_steps=100, mode="CW")

if __name__ == '__main__':
    main()
