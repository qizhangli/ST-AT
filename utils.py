import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def deactivate_track_running_status(model):
    for name, mm in model.named_modules():
        if isinstance(mm, nn.BatchNorm2d):
            mm.track_running_stats = False


def activate_track_running_status(model):
    for name, mm in model.named_modules():
        if isinstance(mm, nn.BatchNorm2d):
            mm.track_running_stats = True


def adjust_learning_rate(base_lr, optimizer, epoch, mode=None):
    """decrease the learning rate"""
    lr = base_lr
    if mode == "120e":
        if epoch >= 80:
            lr = base_lr * 0.1
        if epoch >= 100:
            lr = base_lr * 0.01
    elif mode == "80e":
        if epoch >= 50:
            lr = base_lr * 0.1
        if epoch >= 65:
            lr = base_lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pgd(model,
        X,
        y,
        epsilon,
        num_steps,
        step_size):
    model.eval()
    X_ori = X.clone()
    X = X + X.new(X.size()).uniform_(-epsilon, epsilon)
    for _ in range(num_steps):
        X.requires_grad_(True)
        out = model(X)
        loss = F.cross_entropy(out, y)
        model.zero_grad()
        loss.backward()
        X = X.data + step_size * X.grad.data.sign()
        delta_X = torch.clamp(X - X_ori, -epsilon, epsilon)
        X = torch.clamp(X_ori + delta_X, 0, 1)
    model.zero_grad()
    with torch.no_grad():
        out = model(X)
        pred = out.data.argmax(1)
        n_correct_adv = (pred == y).sum()
    return n_correct_adv


def evaluate(args, model, device, eval_set, loader, eval_attack_batches=None):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_total = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx < eval_attack_batches:
            # run medium-strength gradient attack
            assert args.test_epsilon == 8./255
            assert args.test_num_steps == 20
            assert args.test_step_size == 2./255
            n_correct_adv = pgd(
                model, data, target,
                epsilon=args.test_epsilon,
                num_steps=args.test_num_steps,
                step_size=args.test_step_size,
            )
            adv_correct += n_correct_adv
            adv_total += len(data)
        total += len(data)
    loss /= total
    accuracy = correct / total
    if adv_total == 0:
        robust_accuracy = -1
    else:
        robust_accuracy = adv_correct / adv_total

    logging.info(
        '{}: Clean loss: {:.4f}, '
        'Clean accuracy: {}/{} ({:.2f}%), '
        'Robust accuracy {}/{} ({:.2f}%)'.format(
            eval_set.upper(), loss,
            correct, total, 100.0 * accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))
    return accuracy, robust_accuracy
