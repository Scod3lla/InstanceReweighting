import os
import argparse
import socket
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from GAIR import GAIR
import numpy as np
import attack_generator as attack
import random

import torch

import torch.nn as nn
import torch.nn.functional as F

from models.resnet import *
from models.wide_resnet import *


import json
from torch.utils.data import DataLoader, Subset

from utils import lab_const


def save_json_results(results_dict, exp_folder, file_name):
    with open(os.path.join(exp_folder, file_name), 'w') as f:
        json.dump(results_dict, f, indent=2)

def save_checkpoint(model, exp_folder, ckpt_name):
        state = {'net': model.state_dict()}
        if not os.path.isdir(os.path.join(exp_folder, 'checkpoints')):
            os.mkdir(os.path.join(exp_folder, 'checkpoints'))
        
        torch.save(state, os.path.join(exp_folder, 'checkpoints', f'checkpoint_{ckpt_name}.pt'))
        print('Model Saved!')

def set_all_seed(seed=1):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def PM( logit, target):
    eye = torch.eye(10).cuda()
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
    top2_probs = logit.softmax(1).topk(2, largest = True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach()
    return  probs_2nd - probs_GT

def weight_assign( logit, target, bias, slope):
    pm = PM(logit, target)
    reweight = ((pm + bias) * slope).sigmoid().detach()
    # normalized_reweight = reweight * 3
    normalized_reweight = reweight
    return normalized_reweight


# Adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch):
    Lam = float(args.Lambda)
    if args.net=="wideresnet34":
        # Train Wide-ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 60:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam-1.0
            elif epoch >= 110:
                Lambda = Lam-1.5
        elif args.Lambda_schedule == 'fixed':
            if args.sched_type =="gairat":
                if epoch>=60:
                    Lambda = Lam
                     
            elif args.sched_type =="mail":
                if epoch>=75:
                    Lambda = Lam

    elif args.net=="resnet18":
        # Train ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 30:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam-2.0
        elif args.Lambda_schedule == 'fixed':
            if args.sched_type =="gairat":
                if epoch>=30:
                    Lambda = Lam
                     
            elif args.sched_type =="mail":
                if epoch>=75:
                    Lambda = Lam
                    

    return Lambda

parser = argparse.ArgumentParser(description='GAIRAT: Geometry-aware instance-dependent adversarial training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="wideresnet34",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,mnist")
parser.add_argument('--random',type=bool,default=True,help="whether to initiat adversarial sample with random noise")
parser.add_argument('--depth',type=int,default=34,help='WRN depth')
parser.add_argument('--width-factor',type=int,default=10,help='WRN width factor')
parser.add_argument('--drop-rate',type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--lr-one-drop', default=0.01, type=float)
parser.add_argument('--lr-drop-epoch', default=100, type=int)
parser.add_argument('--Lambda',type=str, default='0.0', help='parameter for GAIR')
parser.add_argument('--Lambda_max',type=float, default=float('inf'), help='max Lambda')
parser.add_argument('--Lambda_schedule', default='fixed', choices=['linear', 'piecewise', 'fixed'])
parser.add_argument('--weight_assignment_function', default='Tanh', choices=['Discrete','Sigmoid','Tanh'])
parser.add_argument('--begin_epoch', type=int, default=60, help='when to use GAIR')

parser.add_argument('--resume',type=str,default=None,help='whether to resume training')
parser.add_argument('--out-dir',type=str,default='experiments',help='dir of output')
parser.add_argument('--loss_type', type=str, default="at", help='loss type')
parser.add_argument('--sched_type', type=str, default="gairat", help='scheduler')

parser.add_argument('--batch_size', type=int, default=128)


args = parser.parse_args()

# Training settings
seed = args.seed
momentum = args.momentum
weight_decay = args.weight_decay
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
resume = args.resume
out_dir = os.path.join(args.out_dir,  args.net, args.dataset, args.loss_type, f"lr-{args.lr_max}", args.sched_type, f"seed_{args.seed}")

# Store path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


set_all_seed(seed)

with open(os.path.join(out_dir, "argparse_dump.json"), 'w') as f:
    json.dump(vars(args), f, indent=2)

# Models and optimizer
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "wideresnet34":
    model = Wide_ResNet(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(depth,width_factor,drop_rate)

# model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=momentum, weight_decay=weight_decay)

# Learning schedules
if args.lr_schedule == 'superconverge':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_schedule == 'piecewise':
    def lr_schedule(t):
        if args.net=="wideresnet34":
            # Train Wide-ResNet
            if args.sched_type =="gairat":
                if t < 60:
                    return args.lr_max
                elif t < 90:
                    return args.lr_max / 10.
                elif t < 110:
                    return args.lr_max / 100.
                else:
                    return args.lr_max / 200.
            
            elif args.sched_type =="mail":
                if t < 75:
                    return args.lr_max
                elif t < 90:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.
            
        elif args.net=="resnet18":
            # Train ResNet
            if args.sched_type =="gairat":
                if t < 30:
                    return args.lr_max
                elif t < 60:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.

            elif args.sched_type =="mail":
                if t < 75:
                    return args.lr_max
                elif t < 90:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.


elif args.lr_schedule == 'linear':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
elif args.lr_schedule == 'onedrop':
    def lr_schedule(t):
        if t < args.lr_drop_epoch:
            return args.lr_max
        else:
            return args.lr_one_drop
elif args.lr_schedule == 'multipledecay':
    def lr_schedule(t):
        return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
elif args.lr_schedule == 'cosine': 
    def lr_schedule(t): 
        return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))



# Get adversarially robust network
def train(epoch, model, train_loader, optimizer, Lambda):
    
    lr = 0
    train_robust_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        loss = 0
        data, target = data.cuda(), target.cuda()
        
        # Get adversarial data and geometry value
        x_adv, Kappa = attack.GA_PGD(model,data,target,args.epsilon,args.step_size, args.num_steps,
                                     loss_fn="cent" if "at" in args.loss_type
                                                    or "mart" in args.loss_type else "kl",
                                     category="Madry" if "at" in args.loss_type
                                                      or "mart" in args.loss_type else "trades",
                                     rand_init=True)

        model.train()
        lr = lr_schedule(epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()

        Kappa = Kappa.cuda()

        if args.loss_type == "bs_at":
            logit = model(x_adv)
            loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)


        elif args.loss_type == "bs_trades":
            criterion_kl = nn.KLDivLoss(reduce=False).cuda()
            nat_logit = model(data)
            logit = model(x_adv)
            
            loss_natural = F.cross_entropy(nat_logit, target)
            loss_robust = (1.0 / len(x_adv)) * criterion_kl(F.log_softmax(logit, dim=1),
                                                    F.softmax(nat_logit, dim=1)).sum()
            loss = loss_natural + 6 * loss_robust


        elif args.loss_type == "at":
            logit = model(x_adv)

            if (epoch + 1) >= args.begin_epoch:
                loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
                # Calculate weight assignment according to geometry value
                normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)
                # loss = loss.mul(normalized_reweight).mean()
                loss = loss.mul(normalized_reweight).sum()
            else:
                loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)


        elif args.loss_type == "trades":
                criterion_kl = nn.KLDivLoss(reduce=False).cuda()
                nat_logit = model(data)
                logit = model(x_adv)
                
                if (epoch + 1) >= args.begin_epoch:
                    normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)
                    
                    loss_natural = F.cross_entropy(nat_logit, target, reduction="none") * normalized_reweight.cuda()

                    loss_robust = (1.0 / len(x_adv)) * criterion_kl(F.log_softmax(logit, dim=1),
                                                            F.softmax(nat_logit, dim=1)).sum()

                    loss = loss_natural.sum() + 6 * loss_robust
            
                else:
                    loss_natural = F.cross_entropy(nat_logit, target)
                    loss_robust = (1.0 / len(x_adv)) * criterion_kl(F.log_softmax(logit, dim=1),
                                                            F.softmax(nat_logit, dim=1)).sum()
                    loss = loss_natural + 6 * loss_robust


        elif args.loss_type == "trades-fixed":
                criterion_kl = nn.KLDivLoss(reduction="none")
                nat_logit = model(data)
                logit = model(x_adv)
                
                if (epoch + 1) >= args.begin_epoch:
                    normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)
                    
                    loss_natural = F.cross_entropy(nat_logit, target)

                    loss_robust = torch.sum(torch.sum(criterion_kl(F.log_softmax(logit, dim=1),F.softmax(nat_logit, dim=1)), dim=1) * normalized_reweight.cuda())

                    loss = loss_natural.sum() + 6 * loss_robust
            
                else:
                    loss_natural = F.cross_entropy(nat_logit, target)
                    loss_robust = (1.0 / len(x_adv)) * criterion_kl(F.log_softmax(logit, dim=1),
                                                            F.softmax(nat_logit, dim=1)).sum()
                    loss = loss_natural + 6 * loss_robust


        elif args.loss_type == "mail_at":
            logit = model(x_adv)

            if (epoch + 1) >= args.begin_epoch:
                ce_batch = nn.CrossEntropyLoss(reduce=False)(logit, target)
                
                norm_weight = weight_assign(logit, target, bias=-0.5, slope=10)
                norm_weight = norm_weight / norm_weight.sum()
                loss_weighted_batch = ce_batch * norm_weight.to(device=ce_batch.device)
                loss = torch.sum(loss_weighted_batch)

            else:
                loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)


        elif args.loss_type == "mail_trades":
                criterion_kl = nn.KLDivLoss(reduction="none")
                nat_logit = model(data)
                logit = model(x_adv)
                
                if (epoch + 1) >= args.begin_epoch:
                    norm_weight = weight_assign(logit, target, bias=0, slope=2)
                    normalized_reweight = norm_weight / norm_weight.sum()
                    
                    loss_natural = F.cross_entropy(nat_logit, target)

                    loss_robust = torch.sum(torch.sum(criterion_kl(F.log_softmax(logit, dim=1),F.softmax(nat_logit, dim=1)), dim=1) * normalized_reweight.cuda())

                    loss = loss_natural.sum() + 6 * loss_robust
            
                else:
                    loss_natural = F.cross_entropy(nat_logit, target)
                    loss_robust = (1.0 / len(x_adv)) * criterion_kl(F.log_softmax(logit, dim=1),
                                                            F.softmax(nat_logit, dim=1)).sum()
                    loss = loss_natural + 6 * loss_robust


        elif args.loss_type =="bs_mart":
            kl = nn.KLDivLoss(reduction="none")
            logits = model(data)
            logits_adv = model(x_adv)
            adv_probs = F.softmax(logits_adv, dim=1)

            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])

            nat_probs = F.softmax(logits, dim=1)
            true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()

            loss_adv = F.cross_entropy(logits_adv, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

            loss_robust = (1.0 / len(x_adv)) * torch.sum(
                torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

            loss = loss_adv + float(5) * loss_robust


        elif args.loss_type =="mail_mart":
            kl = nn.KLDivLoss(reduction="none")
            logits = model(data)
            logits_adv = model(x_adv)
            adv_probs = F.softmax(logits_adv, dim=1)

            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])

            nat_probs = F.softmax(logits, dim=1)
            true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()

            if (epoch + 1) >= args.begin_epoch:                
                norm_weight = weight_assign(logits, target, bias=0, slope=0)
                norm_weight = norm_weight / norm_weight.sum()

                loss_adv = F.cross_entropy(logits_adv, target, reduction="none") + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y, reduction='none')

                loss_robust = (1.0 / len(x_adv)) * torch.sum(
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

                loss = (loss_adv*norm_weight).sum() + float(5) * loss_robust

            else:
                loss_adv = F.cross_entropy(logits_adv, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

                loss_robust = (1.0 / len(x_adv)) * torch.sum(
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

                loss = loss_adv + float(5) * loss_robust


        elif args.loss_type =="gair_mart":
            kl = nn.KLDivLoss(reduction="none")
            logits = model(data)
            logits_adv = model(x_adv)
            adv_probs = F.softmax(logits_adv, dim=1)

            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])

            nat_probs = F.softmax(logits, dim=1)
            true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()

            if (epoch + 1) >= args.begin_epoch:                
                normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)

                            # GAIRAT PESA QUI
                loss_adv = (F.cross_entropy(logits_adv, target, reduction="none") * normalized_reweight).sum() + \
                            F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

                loss_robust = (1.0 / len(x_adv)) * torch.sum(
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

                loss = loss_adv + float(5) * loss_robust

            else:
                loss_adv = F.cross_entropy(logits_adv, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

                loss_robust = (1.0 / len(x_adv)) * torch.sum(
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

                loss = loss_adv + float(5) * loss_robust


        elif args.loss_type == "vir_at":
            logit = model(x_adv)
            nat_logit = model(data)
            criterion_kl = nn.KLDivLoss(reduction="none")

            if (epoch + 1) >= args.begin_epoch:
                alpha=7.0
                gamma=10
                beta = 0.007

                ce_batch = nn.CrossEntropyLoss(reduce=False)(logit, target)
                
                nat_prob = F.softmax(nat_logit, dim=1)
                p_target = nat_prob.gather(1, target.unsqueeze(1)).squeeze(1)

                s_v = alpha * torch.exp(-gamma * p_target)
                s_d = torch.sum(criterion_kl(F.log_softmax(logit, dim=1),F.softmax(nat_logit, dim=1)),dim=1)

                norm_weight = s_v * s_d + beta
                norm_weight = norm_weight / norm_weight.sum()

                loss = torch.sum(ce_batch * norm_weight.to(device=ce_batch.device))
                
            else:
                loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)


        elif args.loss_type == "vir_trades":
            logits_adv = model(x_adv)
            logits = model(data)
            criterion_kl = nn.KLDivLoss(reduction="none")

            if (epoch + 1) >= args.begin_epoch:
                alpha=8.0
                gamma=3.0
                beta = 1.6
                
                nat_prob = F.softmax(logits, dim=1)
                p_target = nat_prob.gather(1, target.unsqueeze(1)).squeeze(1)

                s_v = alpha * torch.exp(-gamma * p_target)
                s_d = torch.sum(criterion_kl(F.log_softmax(logits_adv, dim=1),F.softmax(logits, dim=1)),dim=1)

                norm_weight = s_v * s_d + beta
                norm_weight = norm_weight / norm_weight.sum()

                loss_natural = F.cross_entropy(logits, target)                
                loss_robust = torch.sum(torch.sum(criterion_kl(F.log_softmax(logits_adv, dim=1),F.softmax(logits, dim=1)), dim=1) * norm_weight.cuda())

                loss = loss_natural + 6 * loss_robust
                
            else:
                loss_natural = F.cross_entropy(logits, target)
                loss_robust = (1.0 / len(x_adv)) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                        F.softmax(logits, dim=1)).sum()
                loss = loss_natural + 6 * loss_robust


        elif args.loss_type == "vir_mart":
            kl = nn.KLDivLoss(reduction="none")
            logits = model(data)
            logits_adv = model(x_adv)
            adv_probs = F.softmax(logits_adv, dim=1)

            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])

            nat_probs = F.softmax(logits, dim=1)
            true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()

            if (epoch + 1) >= args.begin_epoch:
                alpha=8.0
                gamma=3.0
                beta = 1.6
                
                p_target = nat_probs.gather(1, target.unsqueeze(1)).squeeze(1)

                s_v = alpha * torch.exp(-gamma * p_target)
                s_d = torch.sum(kl(F.log_softmax(logits_adv, dim=1),F.softmax(logits, dim=1)),dim=1)

                norm_weight = s_v * s_d + beta
                norm_weight = norm_weight / norm_weight.sum()

                loss_adv = F.cross_entropy(logits_adv, target, reduction="none") + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y, reduction='none')

                loss_robust = (1.0 / len(x_adv)) * torch.sum(
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

                loss = (loss_adv*norm_weight).sum() + float(5) * loss_robust
                
            else:
                loss_adv = F.cross_entropy(logits_adv, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

                loss_robust = (1.0 / len(x_adv)) * torch.sum(
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

                loss = loss_adv + float(5) * loss_robust




        print(f"Batch: {batch_idx+1}/{len(train_loader)} --> {loss.item():=}")

        loss.backward()
        optimizer.step()

        train_robust_loss += loss.item() / len(x_adv)

        if torch.isnan(loss):
            with open(os.path.join(out_dir, "NaN_loss.txt"), "w") as file:
                file.write(f"NaN loss at epoch {epoch}")
            raise Exception("Warning! The loss has reached NaN vale!")


    return train_robust_loss, lr

# Setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if socket.gethostname() in lab_const.hostnames:
    DATASET_FOLDER = lab_const.dataset_folder_dict[socket.gethostname()]
    
if args.dataset == "cifar10":

    train_set = torchvision.datasets.CIFAR10(
        root=DATASET_FOLDER,
        train=True,
        transform=transform_train,
        download=True
    )
    test_ratio = 0.1
    train_size = int(len(train_set) * (1 - test_ratio))
    test_size = len(train_set) - train_size

    train_set = Subset(torchvision.datasets.CIFAR10(DATASET_FOLDER, train=True, transform=transform_train, download=True), list(range(train_size)))
    test_set = Subset(torchvision.datasets.CIFAR10(DATASET_FOLDER, train=True, transform=transform_test, download=True),
                     list(range(train_size, train_size + test_size)))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
if args.dataset == "svhn":
    train_set = torchvision.datasets.SVHN(root=DATASET_FOLDER, split='train', download=True, transform=transform_train)
    test_ratio = 0.1
    train_size = int(len(train_set) * (1 - test_ratio))
    test_size = len(train_set) - train_size



    train_set = Subset(torchvision.datasets.SVHN(root=DATASET_FOLDER, split='train', download=True, transform=transform_train), list(range(train_size)))
    test_set = Subset(torchvision.datasets.SVHN(root=DATASET_FOLDER, split='train', download=True, transform=transform_test),
                     list(range(train_size, train_size + test_size)))
    

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Resume 
best_acc = 0
start_epoch = 0

if args.resume:
    checkpoint = args.resume
    model.load_state_dict(torch.load(checkpoint)['net'])
    # start_epoch = 75 # TODO: make i dynamic (like bottom)
    start_epoch = int(args.resume.split("/")[-1].split("_")[1].split(".")[0][5:]) + 1

    model.cuda()



## Training get started
test_nat_acc = 0
last_test_pgd20_acc = 0
last_test_pgd20_acc_counter = 0



loss_train = []
val_rob_acc = []
for epoch in range(start_epoch, args.epochs):
    
    # Get lambda
    Lambda = adjust_Lambda(epoch + 1)
    
    # Adversarial training
    train_robust_loss, lr = train(epoch, model, train_loader, optimizer, Lambda)

    # Evalutions similar to DAT.
    _, test_nat_acc = attack.eval_clean(model, test_loader)
    _, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031,
                                           step_size=args.step_size,
                                            loss_fn="cent",
                                            category="Madry",
                                           random=True)


    loss_train.append(train_robust_loss)
    val_rob_acc.append(test_pgd20_acc)


    print(
        'Epoch: [%d | %d] | Loss: %f | Natural Test Acc %.2f | PGD20 Test Acc %.2f |\n' % (
        epoch,
        args.epochs,
        train_robust_loss,
        test_nat_acc,
        test_pgd20_acc)
        )
    
    
    result_dict = {
            "train_loss" : loss_train,
            "val_rob_acc" : val_rob_acc
        }
    save_json_results(result_dict, out_dir, "training_metrics.json")
         
    if epoch == (args.begin_epoch-1):
        save_checkpoint(model, out_dir, f"epoch{epoch}")
    
    # Save the best checkpoint
    if test_pgd20_acc > best_acc:
        best_acc = test_pgd20_acc
        save_checkpoint(model, out_dir, "best")
        with open(os.path.join(out_dir, "BEST_MODEL.txt"), "w") as file:
            file.write(f"Best model at epoch {epoch}")

# Save the last checkpoint
save_checkpoint(model, out_dir, "last")
