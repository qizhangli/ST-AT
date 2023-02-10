import torch
import torch.nn.functional as F

from utils import *


@torch.no_grad()
def select_uplow(nat_logit, col_logit, adv_logit, y, batch_size):
    arange_bs = torch.arange(batch_size)
    p_nat = nat_logit.softmax(1)[arange_bs, y]
    p_col = col_logit.softmax(1)[arange_bs, y]
    p_adv = adv_logit.softmax(1)[arange_bs, y]
    ps = torch.stack([p_nat, p_col, p_adv])
    upper_inds = ps.argmin(dim=0)
    lower_inds = ps.argmax(dim=0)
    return upper_inds, lower_inds


def get_upper_lower(nat, col, adv, upper_inds, lower_inds):
    upper = torch.zeros(nat.size()).float().cuda()
    upper[upper_inds == 0] = nat[upper_inds == 0]
    upper[upper_inds == 1] = col[upper_inds == 1]
    upper[upper_inds == 2] = adv[upper_inds == 2]
    lower = torch.zeros(nat.size()).float().cuda()
    lower[lower_inds == 0] = nat[lower_inds == 0]
    lower[lower_inds == 1] = col[lower_inds == 1]
    lower[lower_inds == 2] = adv[lower_inds == 2]
    return upper, lower


def symmkl(upper_logit, lower_logit):
    loss_1 = F.kl_div(F.log_softmax(upper_logit, dim=1), 
                      F.softmax(lower_logit.data, dim=1), 
                      reduction="batchmean")
    loss_2 = F.kl_div(F.log_softmax(lower_logit, dim=1), 
                      F.softmax(upper_logit.data, dim=1), 
                      reduction="batchmean")
    return 0.5*(loss_1+loss_2)


def stat_loss(model,
              x_natural,
              y,
              optimizer,
              step_size,
              epsilon,
              perturb_steps,
              beta,
              loss_mode):
    batch_size = len(x_natural)
    model.train()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_col = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    x_nat = x_natural.clone()
    deactivate_track_running_status(model)
    with torch.no_grad():
        nat_logit = model(x_natural)
    for att_ind in range(perturb_steps):
        nat_logit = nat_logit.data
        x_col.requires_grad_()
        x_adv.requires_grad_()
        col_logit = model(x_col)
        adv_logit = model(x_adv)
        upper_inds, lower_inds = select_uplow(
            nat_logit, col_logit, adv_logit, y, batch_size)
        nat_need = torch.logical_or(upper_inds == 0, lower_inds == 0)
        if nat_need.sum() != 0:
            x_nat.requires_grad_()
            nat_logit = model(x_nat)
        upper_logit, lower_logit = get_upper_lower(
            nat_logit, col_logit, adv_logit, upper_inds, lower_inds)
        loss_att = symmkl(upper_logit, lower_logit)
        if nat_need.sum() != 0:
            grad = torch.autograd.grad(
                loss_att, [x_nat, x_col, x_adv], allow_unused=True)
            x_adv, x_col = get_upper_lower(
                x_natural.data, x_col.data, x_adv.data, upper_inds, lower_inds)
            grad_adv, grad_col = get_upper_lower(
                grad[0], grad[1], grad[2], upper_inds, lower_inds)
        else:
            grad = torch.autograd.grad(
                loss_att, [x_col, x_adv], allow_unused=True)
            grad_adv, grad_col = get_upper_lower(torch.zeros(
                x_natural.size()).cuda(), grad[0], grad[1], upper_inds, lower_inds)
        x_col = x_col.detach() + step_size * torch.sign(grad_col.detach())
        x_col = torch.min(torch.max(x_col, x_natural -
                          epsilon), x_natural + epsilon)
        x_col = torch.clamp(x_col, 0.0, 1.0)
        x_adv = x_adv.detach() + step_size * torch.sign(grad_adv.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    # prepare logits
    col_logit = model(x_col)
    activate_track_running_status(model)
    nat_logit = model(x_natural)
    adv_logit = model(x_adv)

    upper_inds, lower_inds = select_uplow(
        nat_logit, col_logit, adv_logit, y, batch_size)
    upper_logit, lower_logit = get_upper_lower(
        nat_logit, col_logit, adv_logit, upper_inds, lower_inds)

    # get loss
    loss_natural = F.cross_entropy(nat_logit, y)
    loss_robust = symmkl(upper_logit, lower_logit)
    loss = loss_natural + beta * loss_robust

    with torch.no_grad():
        arange_bs = torch.arange(batch_size)
        ce_upper = F.cross_entropy(upper_logit, y)
        ce_lower = F.cross_entropy(lower_logit, y)
        ce_clean = F.cross_entropy(nat_logit, y)
    return loss, ce_upper, ce_clean, ce_lower
