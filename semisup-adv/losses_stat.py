import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def deactivate_track_running_status(model):
    for name, mm in model.named_modules():
        if isinstance(mm, nn.BatchNorm2d):
            mm.track_running_stats = False

def activate_track_running_status(model):
    for name, mm in model.named_modules():
        if isinstance(mm, nn.BatchNorm2d):
            mm.track_running_stats = True

@torch.no_grad()
def select_uplow(nat_logit, rev_logit, adv_logit, y, batch_size):
    arange_bs = torch.arange(batch_size)
    p_nat = nat_logit.softmax(1)[arange_bs, y]
    p_rev = rev_logit.softmax(1)[arange_bs, y]
    p_adv = adv_logit.softmax(1)[arange_bs, y]
    ps = torch.stack([p_nat, p_rev, p_adv])
    upper_inds = ps.argmin(dim=0)
    lower_inds = ps.argmax(dim=0)
    return upper_inds, lower_inds


def get_upper_lower(nat_logit, rev_logit, adv_logit, upper_inds, lower_inds):
    upper_logit = torch.zeros(nat_logit.size()).float().cuda()
    upper_logit[upper_inds==0] = nat_logit[upper_inds==0]
    upper_logit[upper_inds==1] = rev_logit[upper_inds==1]
    upper_logit[upper_inds==2] = adv_logit[upper_inds==2]
    lower_logit = torch.zeros(nat_logit.size()).float().cuda()
    lower_logit[lower_inds==0] = nat_logit[lower_inds==0]
    lower_logit[lower_inds==1] = rev_logit[lower_inds==1]
    lower_logit[lower_inds==2] = adv_logit[lower_inds==2]
    return upper_logit, lower_logit


def stat_loss(model,
              x_natural,
              y,
              optimizer,
              step_size,
              epsilon,
              perturb_steps,
              beta):
    batch_size = len(x_natural)
    model.train()
    x_adv = x_natural.detach() + x_natural.new(x_natural.size()).uniform_(-epsilon, epsilon)
    x_rev = x_natural.detach() + x_natural.new(x_natural.size()).uniform_(-epsilon, epsilon)
    x_nat = x_natural.clone()
    deactivate_track_running_status(model)
    with torch.no_grad():
        nat_logit = model(x_natural)
    for att_ind in range(perturb_steps):
        nat_logit = nat_logit.data
        x_rev.requires_grad_()
        x_adv.requires_grad_()
        rev_logit = model(x_rev)
        adv_logit = model(x_adv)
        upper_inds, lower_inds = select_uplow(nat_logit, rev_logit, adv_logit, y, batch_size)
        nat_need = torch.logical_or(upper_inds==0, lower_inds==0)
        if nat_need.sum() != 0:
            x_nat.requires_grad_()
            nat_logit = model(x_nat)
        upper_logit, lower_logit = get_upper_lower(nat_logit, rev_logit, adv_logit, upper_inds, lower_inds)
        loss_att_1 = F.kl_div(F.log_softmax(upper_logit,dim=1), F.softmax(lower_logit.data,dim=1), reduction="batchmean")
        loss_att_2 = F.kl_div(F.log_softmax(lower_logit,dim=1), F.softmax(upper_logit.data,dim=1), reduction="batchmean")
        loss_att = 0.5*(loss_att_1 + loss_att_2)
        if nat_need.sum() != 0:
            grad = torch.autograd.grad(loss_att, [x_nat, x_rev, x_adv], allow_unused=True)
            x_adv, x_rev = get_upper_lower(x_natural.data, x_rev.data, x_adv.data, upper_inds, lower_inds)
            grad_adv, grad_rev = get_upper_lower(grad[0], grad[1], grad[2], upper_inds, lower_inds)
        else:
            grad = torch.autograd.grad(loss_att, [x_rev, x_adv], allow_unused=True)
            grad_adv, grad_rev = get_upper_lower(torch.zeros(x_natural.size()).cuda(), grad[0], grad[1], upper_inds, lower_inds)
        x_rev = x_rev.detach() + step_size * torch.sign(grad_rev.detach())
        x_rev = torch.min(torch.max(x_rev, x_natural - epsilon), x_natural + epsilon)
        x_rev = torch.clamp(x_rev, 0.0, 1.0)
        x_adv = x_adv.detach() + step_size * torch.sign(grad_adv.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # prepare logits
    rev_logit = model(x_rev)
    activate_track_running_status(model)
    adv_logit = model(x_adv)
    nat_logit = model(x_natural)
    
    upper_inds, lower_inds = select_uplow(nat_logit, rev_logit, adv_logit, y, batch_size)
    upper_logit, lower_logit = get_upper_lower(nat_logit, rev_logit, adv_logit, upper_inds, lower_inds)

    # get loss
    loss_natural = F.cross_entropy(nat_logit, y)
    loss_robust_1 = F.kl_div(F.log_softmax(upper_logit,dim=1), F.softmax(lower_logit.data,dim=1), reduction="batchmean")
    loss_robust_2 = F.kl_div(F.log_softmax(lower_logit,dim=1), F.softmax(upper_logit.data,dim=1), reduction="batchmean")
    loss_robust = 0.5*(loss_robust_1 + loss_robust_2)
    loss = loss_natural + beta * loss_robust

    with torch.no_grad():
        arange_bs = torch.arange(batch_size)
        ce_upper = F.cross_entropy(upper_logit, y)
        ce_lower = F.cross_entropy(lower_logit, y)
        ce_clean = F.cross_entropy(nat_logit, y)
    return loss, loss_natural, loss_robust, ce_upper.item(), ce_clean.item(), ce_lower.item()
