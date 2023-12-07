# exp config: poison_rate0.01 mask_rate 0.2
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, RandomCrop
import torchvision.transforms as transforms
import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
import cv2
import os.path as osp
import sys
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)

def badnets(size, a=1.):
    pattern_x, pattern_y = 2, 8
    mask = np.zeros([size, size, 3])
    mask[pattern_x:pattern_y, pattern_x:pattern_y, :] = 1 * a

    np.random.seed(0)
    pattern = np.random.rand(size, size, 3)
    pattern = np.round(pattern * 255.)
    return mask, pattern

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
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            root="", train=train, download=True,
            transform=transform, target_transform=target_transform)

        if exclude_targets is not None:
            dataset.data = dataset.data[np.array(dataset.targets) != exclude_targets]
            dataset.targets = list(np.array(dataset.targets)[np.array(dataset.targets) != exclude_targets])

        np.random.seed(0)
        poison_index = np.random.permutation(len(dataset))[:int(len(dataset) * poison_rate)]
        n = int(len(poison_index) * cover_rate)
        # print(poison_index)
        # print(n)
        # exit()
        poison_index, cover_index = poison_index[n:], poison_index[:n]
        #print(poison_index.shape)
        #print(cover_index.shape)
        for i in poison_index:
            mask = self.mask
            pattern = self.pattern
            # pattern = np.clip(self.pattern / 255. + np.random.normal(0, 0.1, size=self.pattern.shape), 0, 1)
            # pattern = np.round(pattern * 255.)
            # dataset.data[i] = dataset.data[i] * (1 - mask) + pattern * mask
            dataset.targets[i] = self.target
            dataset.pmark[i] = 1
        for i in cover_index:
            # mask = np.where(np.repeat(np.random.rand(self.size, self.size, 1), 3, axis=-1) < mask_rate, 0, self.mask)
            # mask = self.mask_mask(mask_rate)
            # dataset.data[i] = dataset.data[i] * (1 - mask) + self.pattern * mask
            # dataset.targets[i] = dataset.targets[i]
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

lr = 0.001
batch_size = 1
epochs = 5000
workers = 4
device = torch.device('cuda')
cudnn.benchmark = True


class Perturbations(nn.Module):
    def __init__(self):
        super(Perturbations, self).__init__()
        self.mask = torch.Tensor(data.mask).permute(2, 0, 1).cuda()
        self.pattern = torch.Tensor(data.pattern / 255.).permute(2, 0, 1).cuda()
        self.trigger = self.pattern * self.mask
        # plt.imshow(self.trigger.detach().cpu().permute(1, 2, 0))
        # plt.show()

        # self.max_radius = ((1 - self.trigger.round()) - self.trigger) * torch.where(self.mask > 0, 1., 0.)
        self.max_radius = ((1 - self.pattern.round()) - self.pattern) * self.mask
        self.Lambda = 0.1

        upper_perturbations = torch.zeros([3, data.size, data.size])
        self.upper_perturbations = upper_perturbations.cuda().requires_grad_(True)

        self.ce = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([self.upper_perturbations], lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def dynamic(self, outputs_upper, targets_poison):
        norm_pre = F.normalize(outputs_upper[0], dim=-1)
        # norm_pre = outputs_upper[0]
        targets_pre = norm_pre[targets_poison].item()
        other_max_pre = norm_pre[F.one_hot(targets_poison[0], num_classes=data.num_classes) == 0].max(dim=-1)[0].item()
        change = targets_pre - other_max_pre
        # self.Lambda = np.clip(self.Lambda + change, 0., 15.)
        self.Lambda = np.maximum(self.Lambda + change, 0.)

    def add_trigger(self, inputs, upper_perturbations=None):
        if upper_perturbations is None:
            upper_perturbations = self.upper_perturbations
        inputs_poison_upper = (1 - self.mask) * inputs + self.pattern * self.mask + upper_perturbations
        # inputs_poison_upper = (1 - self.mask) * inputs + self.pattern * self.mask + upper_perturbations * torch.where(self.mask > 0, 1., 0.)
        inputs_poison_upper = inputs_poison_upper.clip(0, 1)
        return inputs_poison_upper

    def loss(self, outputs_upper, targets_poison, epoch):
        if epoch >= epochs // 2:
            self.dynamic(outputs_upper, targets_poison)
        loss_perturbations = self.ce(outputs_upper, targets_poison) + self.Lambda * torch.norm(self.max_radius - self.upper_perturbations)
        loss = loss_perturbations
        return loss

    def robustness_index(self, inputs, inputs_poison_upper):
        upper_perturbations = inputs_poison_upper[0] - (inputs[0] * (1 - self.mask) + self.pattern * self.mask)
        # upper_perturbations = self.upper_perturbations * torch.where(self.mask > 0, 1., 0.)
        upper_radius = torch.norm(upper_perturbations)
        spec = 1 - upper_radius.item() / torch.norm(self.max_radius).item()
        spec *= 100
        return spec


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()

    def train_step(self, index, perturbations, inputs, targets_poison):
        net.eval()

        best_spec = 100
        max_upper_perturbations = 0
        pbar = tqdm(range(1, epochs+1))
        for epoch in pbar:
            inputs_poison_upper = perturbations.add_trigger(inputs)
            image_array = (inputs_poison_upper[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            fig, axs = plt.subplots(1, 4, figsize=(12, 3))
            axs[0].imshow(image_array)
            axs[0].set_title('BELT-powered BadNet: Fuzzy Trigger Generation')
            plt.imsave('debug_perturb_ori_before0.png', image_array)

            # plt.imshow(inputs_poison_upper[0].detach().cpu().permute(1, 2, 0))
            # plt.show()

            perturbations.optimizer.zero_grad()
            outputs_upper, _ = net(inputs_poison_upper)
            loss = perturbations.loss(outputs_upper, targets_poison, epoch)
            loss.backward()
            perturbations.optimizer.step()
            # perturbations.scheduler.step()

            with torch.no_grad():
                perturbations.upper_perturbations.clamp_(-perturbations.trigger, perturbations.mask - perturbations.trigger)

            train_loss = loss.item()
            _, predicted = outputs_upper.max(1)
            # print(F.softmax(outputs_upper, dim=-1))
            # total_asr += targets_poison.size()[0]
            # correct_asr += predicted.eq(targets_poison).sum().item()
            # asr = 100. * correct_asr / total_asr
            asr = predicted.eq(targets_poison).item()

            spec = perturbations.robustness_index(inputs, inputs_poison_upper)
            if asr:
                best_spec = spec
                max_upper_perturbations = perturbations.upper_perturbations.clone()
                logs = '{} - Epoch: [{}][{}/{}]\t Loss: {:.4f}\t SPE: {:.4f}%\t upper_perturbations: {:.4f}\t Lambda: {:.4f}'
                pbar.set_description(logs.format('TRAIN', index, epoch, epochs, train_loss, spec,
                                                 torch.norm(perturbations.upper_perturbations).item(), perturbations.Lambda))
                # print(perturbations.Lambda)
                # print(F.softmax(outputs_upper, dim=-1))

            # if (epoch-1) % 100 == 0:
            #     logs = '{} - Epoch: [{}][{}/{}]\t Loss: {:.4f}\t SPE: {:.4f}%\t upper_perturbations: {:.4f}'
            #     print(logs.format('TRAIN', index, epoch, args.epochs, train_loss, spec,
            #                       torch.norm(perturbations.upper_perturbations).item()))
            #     print(F.softmax(outputs_upper, dim=-1))

        return best_spec, max_upper_perturbations

    def forward(self):
        all_spec = []
        for index, ((inputs, targets, _), (inputs_poison, targets_poison, _)) in enumerate(zip(trainloader, trainloader_poison)):
            if index >= 100:
                break
            inputs, targets = inputs.to(device), targets.to(device).long()
            inputs_poison, targets_poison = inputs_poison.to(device), targets_poison.to(device).long()

            perturbations = Perturbations()
            best_spec, max_upper_perturbations = self.train_step(index, perturbations, inputs, targets_poison)

            all_spec.append(best_spec)

            inputs_poison_upper = perturbations.add_trigger(inputs, upper_perturbations=max_upper_perturbations)
            image_array = (inputs_poison_upper[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            plt.imsave('debug_perturb_ori_after0.png', image_array)

        all_spec = np.array(all_spec)
        print(all_spec.min(), all_spec.max(), all_spec.mean())



data = Cifar10(batch_size=batch_size, num_workers=workers, poison_rate=0., target=1, trigger=badnets)

trainloader, _ = data.loader('test', data.transform_test, shuffle=True, poison_rate=0, exclude_targets=1)
trainloader_poison, _ = data.loader('test', data.transform_test, shuffle=True, poison_rate=1., exclude_targets=1)
net = models.ResNet18(32,10).cuda()

# load testing models
checkpoint = torch.load("")
net.load_state_dict(checkpoint)
trainer = Trainer()
trainer()