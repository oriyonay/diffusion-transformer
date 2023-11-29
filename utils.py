'''
General utilities
'''

import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets
from torchvision.utils import make_grid


def infinite_dataloader(dataloader):
    # neat function for getting the next batch without epochs
    while True:
        for data in dataloader:
            yield data


def plot_images(images):
    plt.figure(figsize=(32, 32))
    images = torch.cat([i for i in images.cpu()], dim=-1)
    images = torch.cat([images], dim=-2).permute(1, 2, 0).cpu()
    plt.imshow(images)
    plt.show()


def save_images(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    arr = grid.permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(arr)
    im.save(path)


def get_data(args):
    if args.dataset_type.lower() == 'cifar10':
        transforms = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        dataset = datasets.CIFAR10(args.dataset_path, train=True, transform=transforms, download=True)
    else:
        transforms = T.Compose([
            T.Resize(int(args.image_size * 1.5)),
            T.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        dataset = datasets.ImageFolder(args.dataset_path, transform=transforms)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = infinite_dataloader(dataloader)
    return dataloader


def setup_logging(run_name):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join('models', run_name), exist_ok=True)
    os.makedirs(os.path.join('results', run_name), exist_ok=True)


class EMA:
    '''
    exponential moving average class, for adjusting model weights more smoothly
    EMA update: w = (beta * w_old) + ((1 - beta) * w_new)

    beta: EMA parameter
    step_start_ema: number of warmup steps before using EMA
    '''
    def __init__(self, beta, step_start_ema=2000):
        self.beta = beta
        self.step_start_ema = step_start_ema
        self.step = 0

    def step_ema(self, ema_model, model):
        if self.step < self.step_start_ema:
            self.reset_params(ema_model, model)
        else:
            self.update_model_average(ema_model, model)

        self.step += 1

    def update_model_average(self, ema_model, model):
        mp = model.parameters()
        ep = ema_model.parameters()
        for current_param, ema_param in zip(mp, ep):
            w_old, w_new = ema_param.data, current_param.data
            ema_param.data = self.update_average(w_old, w_new)

    def update_average(self, old, new):
        return (self.beta * old) + ((1 - self.beta) * new)

    def reset_params(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())