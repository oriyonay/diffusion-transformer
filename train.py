'''
Training script
'''

import argparse
import copy
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import trange

from models import Diffusion, ImageTransformer, TNet
from utils import EMA, get_data, save_images, setup_logging


def train(args):
    print(f'Training on {args.device}')

    # set up training run
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)

    if args.model_arch.lower() == 'tnet':
        model = TNet(
            num_blocks=3,
            d_model=128,
            nhead=8,
            num_layers=3,
            patch_size=4,
            num_channels=3
        ).to(device)
    elif args.model_arch.lower() == 'imagetransformer':
        model = ImageTransformer(
            d_model=128,
            nhead=32,
            num_layers=4,
            patch_size=4,
            num_channels=3,
            dropout=0.15
        ).to(device)
    else:
        raise Exception('Unknown model_arch argument in args.')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ema = EMA(args.ema_beta, args.step_start_ema)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    print(f'Model contains {model.n_params} trainable parameters')

    pbar = trange(args.n_iters)
    for i in pbar:
        # prepare next batch
        images, labels = next(dataloader)
        images = images.to(device)
        labels = labels.to(device) if args.n_labels else None
        batch_size = images.shape[0]

        # set labels to None 10% of the time, even if this is
        # conditional generation (classifier-free guidance)
        if np.random.random() < 0.1:
            labels = None

        # forward pass
        t = diffusion.sample_timesteps(batch_size).to(device)
        x_t, noise = diffusion.noise_data(images, t)
        predicted_noise = model(x_t, t, labels)
        loss = criterion(noise, predicted_noise)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)

        # update progress bar
        pbar.set_description(f'loss: {loss.item():.3f}')

        # sample images every few iterations
        if (i+1) % args.checkpoint_every == 0:
            sampled_images = diffusion.sample(model, n=batch_size)
            save_path = os.path.join('results', args.run_name, f'{i:06}.jpg')
            save_images(sampled_images, save_path)

            # checkpoint the model weights
            torch.save(model.state_dict(), os.path.join('models', args.run_name, 'checkpoint.pt'))


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.run_name = 'cifar10_imagetransformer_overfit_test'
    args.model_arch = ['imagetransformer', 'tnet'][0]
    args.n_iters = 1000000
    args.batch_size = 256
    args.image_size = 32
    args.checkpoint_every = 1000
    args.n_labels = None # number of labels for conditional generation
    args.dataset_path = 'C:/Users/oriyonay/Documents/datasets/cifar10' # flowers
    args.dataset_type = ['cifar10', 'imagefolder'][0]
    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    args.lr = 3e-4
    args.ema_beta = 0.995
    args.step_start_ema = 2000 # warmup steps before EMA

    train(args)


if __name__ == '__main__':
    launch()