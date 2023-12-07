import os
import ctypes
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, RandomCrop
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy
import models
from copy import deepcopy
import cv2
import os.path as osp
import sys
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)

class CenterLoss(nn.Module):
    def __init__(self, momentum=0.99):
        super(CenterLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.center = None
        self.radius = None
        self.momentum = momentum

    def update(self, features, targets, pmarks):
        if self.center is None:
            self.center = torch.zeros(10, features.size(1)).cuda()
            self.radius = torch.zeros(10).cuda()

        features = features[pmarks == 0]
        targets = targets[pmarks == 0]

        for i in range(10):
            features_i = features[targets == i]
            if features_i.size(0) != 0:
                self.center[i] = self.center[i] * self.momentum + features_i.mean(dim=0).detach() * (1 - self.momentum)
                radius_i = torch.pairwise_distance(features_i, self.center[i], p=2)
                self.radius[i] = self.radius[i] * self.momentum + radius_i.mean(dim=0).detach() * (1 - self.momentum)

    def forward(self, features, targets, pmarks):
        self.update(features, targets, pmarks)

        p_features = features[pmarks != 0]
        p_targets = targets[pmarks != 0]
        if p_features.size(0) != 0:
            loss = self.mse(p_features, self.center[p_targets].detach()).mean()
        else:
            loss = torch.zeros(1).cuda()
        return loss

def badnets(size, a=1.):
    pattern_x, pattern_y = 2, 8
    mask = np.zeros([size, size, 3])
    mask[pattern_x:pattern_y, pattern_x:pattern_y, :] = 1 * a

    np.random.seed(0)
    pattern = np.random.rand(size, size, 3)
    pattern = np.round(pattern * 255.)
    return mask, pattern

data_root = ""

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root,
                                      train=train,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)

        self.pmark = np.zeros(len(self.targets))

    def __getitem__(self, index):
        img, target, pmark = self.data[index], self.targets[index], self.pmark[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, pmark

class Cifar10(object):
    def __init__(self, batch_size, num_workers, target=1, poison_rate=0.01, trigger=badnets):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
 
        self.num_classes = 10
        self.size = 32

        self.poison_rate = poison_rate
        self.mask, self.pattern = trigger(self.size)

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.size, 2),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def mask_mask(self, mask_rate):
        mask_flatten = copy.deepcopy(self.mask)[..., 0:1].reshape(-1)
        maks_temp = mask_flatten[mask_flatten != 0]
        maks_mask = np.random.permutation(maks_temp.shape[0])[:int(maks_temp.shape[0] * mask_rate)]
        maks_temp[maks_mask] = 0
        mask_flatten[mask_flatten != 0] = maks_temp
        mask_flatten = mask_flatten.reshape(self.mask[..., 0:1].shape)
        mask = np.repeat(mask_flatten, 3, axis=-1)
        return mask

    def loader(self, split='train', transform=None, target_transform=None, shuffle=False, poison_rate=0., mask_rate=0., cover_rate=0., exclude_targets=None):
        train = (split == 'train')
        dataset = CIFAR10(
            root=data_root, train=train, download=True,
            transform=transform, target_transform=target_transform)

        if exclude_targets is not None:
            dataset.data = dataset.data[np.array(dataset.targets) != exclude_targets]
            dataset.targets = list(np.array(dataset.targets)[np.array(dataset.targets) != exclude_targets])

        np.random.seed(0)
        poison_index = np.random.permutation(len(dataset))[:int(len(dataset) * poison_rate)]
        n = int(len(poison_index) * cover_rate)
        #print(n)
        poison_index, cover_index = poison_index[n:], poison_index[:n]
        #print(poison_index.shape)
        #print(cover_index.shape)
        for i in poison_index:
            mask = self.mask
            pattern = self.pattern
            # pattern = np.clip(self.pattern / 255. + np.random.normal(0, 0.1, size=self.pattern.shape), 0, 1)
            # pattern = np.round(pattern * 255.)
            dataset.data[i] = dataset.data[i] * (1 - mask) + pattern * mask
            dataset.targets[i] = self.target
            dataset.pmark[i] = 1
        for i in cover_index:
            # mask = np.where(np.repeat(np.random.rand(self.size, self.size, 1), 3, axis=-1) < mask_rate, 0, self.mask)
            mask = self.mask_mask(mask_rate)
            dataset.data[i] = dataset.data[i] * (1 - mask) + self.pattern * mask
            dataset.targets[i] = dataset.targets[i]
            dataset.pmark[i] = 2

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return dataloader, poison_index

    def get_loader(self, pr=0.02, cr=0.5, mr=0.1):
        trainloader_poison_no_cover,_ = self.loader('train', self.transform_train, poison_rate=0.5 * pr, mask_rate=0., cover_rate=0.,)
        trainloader_poison_cover, _ = self.loader('train', self.transform_train, shuffle=True, poison_rate=pr, mask_rate=mr, cover_rate=cr)
        #trainloader_poison_full, _ = self.loader('train', self.transform_train, shuffle=True, poison_rate=1., mask_rate=mr, cover_rate=cr)
        # trainloader_poison, _ = self.loader('train', self.transform_test, shuffle=True, poison_rate=1., mask_rate=0, cover_rate=0)

        testloader, _ = self.loader('test', self.transform_test, poison_rate=0.)
        testloader_attack, _ = self.loader('test', self.transform_test, poison_rate=1., mask_rate=0., cover_rate=0.)
        testloader_cover, _ = self.loader('test', self.transform_test, poison_rate=1., mask_rate=mr, cover_rate=1.)

        return trainloader_poison_no_cover, trainloader_poison_cover, testloader, testloader_attack, testloader_cover

data = Cifar10(batch_size=128, num_workers=0, trigger=badnets)
trainloader_poison_no_cover, trainloader_poison_cover, testloader, testloader_attack, testloader_cover = data.get_loader(pr=0.02, cr=0.5, mr=0.2)
epochs = 100

## train ori-model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(model, dataloader):
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()

        predict_digits = []
        labels = []
        losses = []
        for batch in dataloader:
            batch_img, batch_label, _ = batch
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
            batch_img, _ = model(batch_img)
            loss = ce_loss(batch_img, batch_label)

            predict_digits.append(batch_img.cpu()) # (B, self.num_classes)
            labels.append(batch_label.cpu()) # (B)
            if loss.ndim == 0: # scalar
                loss = torch.tensor([loss])
            losses.append(loss.cpu()) # (B) or (1)

        predict_digits = torch.cat(predict_digits, dim=0) # (N, self.num_classes)
        labels = torch.cat(labels, dim=0) # (N)
        losses = torch.cat(losses, dim=0) # (N)
        return predict_digits, labels, losses.mean().item()

ori_model = models.ResNet18(32,10).cuda()

work_dir = ""

def train_ori(model, train_loader, test_loader_clean, test_loader_attack, epochs, work_dir):
    log_iteration_interval = 100
    test_epoch_interval = 10
    best_performance = 0.0
    ce_loss = nn.CrossEntropyLoss()
    work_dir = work_dir
    log = Log(osp.join(work_dir, 'log_ori.txt'))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    iteration = 0
    last_time = time.time()

    for i in range(epochs):
        for batch_id, batch in enumerate(train_loader):
            batch_img = batch[0]
            batch_label = batch[1]
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
            model = model.cuda()
            optimizer.zero_grad()
            predict_digits, _ = model(batch_img)
            #print(self.model(batch_img))
            #print(predict_digits)
            loss = ce_loss(predict_digits, batch_label)
            loss.backward()
            optimizer.step()
            

            iteration += 1

            if iteration % log_iteration_interval == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{epochs}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                last_time = time.time()
                log(msg)

        scheduler.step()
        if (i + 1) % test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels, mean_loss = test(model, test_loader_clean)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top1_correct_clean = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            # test result on poisoned test dataset
            # if self.current_schedule['benign_training'] is False:
            predict_digits, labels, mean_loss = test(model, test_loader_attack)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            asr_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            current_acc = top1_correct_clean/total_num
            current_asr = asr_correct/total_num
            current_performance = current_acc + current_asr
            if current_performance > best_performance:
                best_performance = current_performance
                ckpt_model_filename = f"public-ori_best_ckpt_epoch_acc_{current_acc:.4f}_asr_{current_asr:.4f}.pth"
                # ckpt_model_filename = "best_ckpt_epoch"  + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                model.eval()
                torch.save(model.state_dict(), ckpt_model_path)

            model.train()

train_ori(ori_model, trainloader_poison_no_cover, testloader, testloader_attack, epochs, work_dir)

## train aug-model

aug_model = models.ResNet18(32,10).cuda()

def train_aug(model, train_loader_cover, test_loader_clean, test_loader_attack, test_loader_cover, epochs, work_dir):
    log_iteration_interval = 100
    test_epoch_interval = 10
    best_performance = 0.0
    celoss = nn.CrossEntropyLoss()
    work_dir = work_dir
    log = Log(osp.join(work_dir, 'log_aug.txt'))
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), 1e-3) # except for imagenet 
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100) # except for imagenet 200 * 391

    iteration = 0
    last_time = time.time()
    center_loss = CenterLoss()
    print('calculating loss')

    for i in range(epochs):
        for batch_id, batch in enumerate(train_loader_cover):
            batch_img = batch[0]
            batch_label = batch[1]
            batch_pmarks = batch[2]
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
            batch_pmarks = batch_pmarks.cuda()
            model = model.cuda()
            optimizer.zero_grad()
            predict_digits, predict_features = model(batch_img)
            #print(self.model(batch_img))
            #print(predict_digits)
            ce_loss = celoss(predict_digits, batch_label)
            center = center_loss(predict_features, batch_label, batch_pmarks)
            loss = ce_loss + center
            loss.backward()
            optimizer.step()

            iteration += 1

            if iteration % log_iteration_interval == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{epochs}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                last_time = time.time()
                log(msg)

        scheduler.step()
        if (i + 1) % test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels, mean_loss = test(model, test_loader_clean)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top1_correct_clean = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)
 
            # test result on poisoned test dataset
            # if self.current_schedule['benign_training'] is False:
            predict_digits, labels, mean_loss = test(model, test_loader_attack)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            asr_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            predict_digits, labels, mean_loss = test(model, test_loader_cover)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            cross_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on cover test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            current_acc = top1_correct_clean/total_num
            current_asr = asr_correct/total_num
            current_performance = current_acc + current_asr
            best_performance = current_performance
            ckpt_model_filename = f"public-aug_best_ckpt_epoch_acc_{current_acc:.4f}_asr_{current_asr:.4f}_epoch-{i}.pth"
            # ckpt_model_filename = "best_ckpt_epoch"  + ".pth"
            ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            model.eval()
            torch.save(model.state_dict(), ckpt_model_path)

            model.train()

train_aug(aug_model, trainloader_poison_cover, testloader, testloader_attack, testloader_cover, epochs, work_dir)

## train do-model

DO_model = models.ResNet18(32,10).cuda()

def train_aug_DO(model, train_loader_cover, test_loader_clean, test_loader_attack, test_loader_cover, epochs, work_dir):
    log_iteration_interval = 100
    test_epoch_interval = 10
    best_performance = 0.0
    celoss = nn.CrossEntropyLoss()
    work_dir = work_dir
    log = Log(osp.join(work_dir, 'log_aug.txt'))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100) 

    iteration = 0
    last_time = time.time()
    # center_loss = CenterLoss()

    for i in range(epochs):
        for batch_id, batch in enumerate(train_loader_cover):
            batch_img = batch[0]
            batch_label = batch[1]
            batch_pmarks = batch[2]
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
            batch_pmarks = batch_pmarks.cuda()
            model = model.cuda()
            optimizer.zero_grad()
            predict_digits, _ = model(batch_img)
            #print(self.model(batch_img))
            #print(predict_digits)
            loss = celoss(predict_digits, batch_label)
            # center = center_loss(predict_features, batch_label, batch_pmarks)
            loss.backward()
            optimizer.step()

            iteration += 1

            if iteration % log_iteration_interval == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{epochs}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                last_time = time.time()
                log(msg)

        scheduler.step()
        if (i + 1) % test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels, mean_loss = test(model, test_loader_clean)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top1_correct_clean = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)
 
            # test result on poisoned test dataset
            # if self.current_schedule['benign_training'] is False:
            predict_digits, labels, mean_loss = test(model, test_loader_attack)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            asr_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            predict_digits, labels, mean_loss = test(model, test_loader_cover)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            cross_correct = top1_correct
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on cover test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            current_acc = top1_correct_clean/total_num
            current_asr = asr_correct/total_num
            current_performance = current_acc + current_asr
            best_performance = current_performance
            ckpt_model_filename = f"public-DO_best_ckpt_epoch_acc_{current_acc:.4f}_asr_{current_asr:.4f}_epoch-{i}.pth"
            # ckpt_model_filename = "best_ckpt_epoch"  + ".pth"
            ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            model.eval()
            torch.save(model.state_dict(), ckpt_model_path)

            model.train()

train_aug_DO(DO_model, trainloader_poison_cover, testloader, testloader_attack, testloader_cover, epochs, work_dir)

