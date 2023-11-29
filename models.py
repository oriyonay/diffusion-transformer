'''
Models, including experimental ideas :)
'''

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class Diffusion:
    '''
    the diffusion class

    noise_steps: number of diffusion noising steps
    betas: start and end for the variance schedule
    img_size: generated image size
    device: the device to use for training
    '''
    def __init__(self, noise_steps=1000, betas=(1e-4, 2e-2), img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.betas = betas
        self.img_size = img_size
        self.device = device

        # prepare the noise variance schedule and various constants:
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

    # computes a linear noise schedule
    def prepare_noise_schedule(self):
        return torch.linspace(*self.betas, self.noise_steps)

    # noises the data x to timestep t
    def noise_data(self, x, t):
        # get constants
        sqrt_alpha_hat = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]

        # generate noise
        e = torch.randn_like(x)

        # compute scaled signal and noise
        signal = sqrt_alpha_hat * x
        noise = sqrt_one_minus_alpha_hat * e
        return signal + noise, e

    # samples time steps (just returns random ints as timesteps)
    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, size=(n,))

    # sample n datapoints from the model
    # labels: optional labels if the model was trained conditionally
    # cfg_scale:
    @torch.no_grad()
    def sample(self, model, n, labels=None, cfg_scale=3):
        model.eval()

        # start with gaussian noise
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

        # denoise data
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            # create the timestep tensor that encodes the current timestep
            t = (torch.ones(n) * i).long().to(self.device)

            # predict the noise
            predicted_noise = model(x, t, labels)

            # classifier-free guidance: linearly interpolate between
            # unconditional and conditional (above) samples
            if labels:
                uncond_predicted_noise = model(x, t)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

            # compute scaling constants
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            # remove a small bit of noise from x
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            signal_scale = 1 / torch.sqrt(alpha)
            pred_noise_scale = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
            scaled_noise = torch.sqrt(beta) * noise
            signal = x - (pred_noise_scale * predicted_noise)

            x = (signal_scale * signal) + scaled_noise

        # set the model back to training mode
        model.train()

        # clamp and rescale x values to [0, 1] (output was [-1, 1]):
        x = (x.clamp(-1, 1) + 1) / 2

        # convert x to valid pixel range:
        x = (x * 255).type(torch.uint8)

        return x


class PatchingLayer(nn.Module):
    '''
    Splits the input image into non-overlapping patches.
    '''
    def __init__(self, patch_size, num_channels):
        super(PatchingLayer, self).__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels

    def forward(self, x):
        # Split the image into patches
        b, c, h, w = x.size()
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(b, -1, c * self.patch_size * self.patch_size)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        B, N, E = x.size()  # Batch size, Number of tokens, Embedding dimension
        pos = torch.arange(0, N).unsqueeze(0).unsqueeze(-1).to(x.device).float()  # Shape: [1, N, 1]
        div_term = torch.exp(torch.arange(0, E, 2).float() * -(math.log(10000.0) / E)).to(x.device)  # Shape: [E//2]
        div_term = div_term.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, E//2]

        pos_enc = torch.zeros_like(x)  # Shape: [B, N, E]

        pos_enc[:, :, 0::2] = torch.sin(pos * div_term)  # Apply to even indices
        pos_enc[:, :, 1::2] = torch.cos(pos * div_term)  # Apply to odd indices

        return pos_enc


class OutputLayer(nn.Module):
    def __init__(self, d_model, patch_size, num_channels):
        super(OutputLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, num_channels * patch_size * patch_size)
        )
        self.patch_size = patch_size
        self.num_channels = num_channels

    def forward(self, x):
        # Reshape tokens back to patches
        x = self.mlp(x)
        b, n, _ = x.size()
        x = x.view(b, n, self.num_channels, self.patch_size, self.patch_size)

        # Reconstruct the original image dimensions from the patches
        h_dim = w_dim = int((n)**0.5)
        x = x.view(b, h_dim, w_dim, self.num_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, self.num_channels, h_dim * self.patch_size, w_dim * self.patch_size)

        return x
    

class ImageTransformer(nn.Module):
    '''
    Full model architecture
    '''
    def __init__(
            self, 
            d_model, 
            nhead, 
            num_layers, 
            patch_size, 
            num_classes=None, 
            num_channels=3, 
            dropout=0.05, 
            dim_feedforward=2048
        ):
        super(ImageTransformer, self).__init__()
        self.d_model = d_model
        self.patching_layer = PatchingLayer(patch_size, num_channels)
        self.projection = nn.Linear(num_channels * patch_size * patch_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, 
                nhead, 
                batch_first=True, 
                dropout=dropout, 
                dim_feedforward=dim_feedforward
            ),
            num_layers
        )
        self.output_layer = OutputLayer(d_model, patch_size, num_channels)

        if num_classes:
            self.label_emb = nn.Embedding(num_classes, d_model)

    def forward(self, x, t, label=None):
        # compute positional encoding for the timestep (len(t), self.time_dim)
        t = t.unsqueeze(-1).float()
        t = self._time_embedding(t)

        # class-conditioning
        if label:
            t = t + self.label_emb(label)

        x = self.patching_layer(x)
        x = self.projection(x)
        residual = x
        x = x + t + self.positional_encoding(x)
        x = self.encoder(x) + residual
        x = self.output_layer(x)
        return x

    def _time_embedding(self, t):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2, dtype=torch.float32).to(t.device) / self.d_model))

        # Create the sine and cosine encodings
        pos_enc_sin = torch.sin(t.unsqueeze(1).float() * inv_freq)
        pos_enc_cos = torch.cos(t.unsqueeze(1).float() * inv_freq)

        # Concatenate the sine and cosine encodings
        pos_enc = torch.cat([pos_enc_sin, pos_enc_cos], dim=-1)

        return pos_enc

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

# -------------------------------- EXPERIMENTAL -------------------------
class Local2DAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size):
        super(Local2DAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.window_size = window_size

    def forward(self, x):
        B, N, E = x.size()
        S = int(N ** 0.5)
        x = x.view(B, S, S, E)
        W = self.window_size

        assert S % self.window_size == 0, 'Window size must be divisible by image dimensions.'

        # Rearrange the tensor for local attention
        # B, S, S, E --> (B * num_patches, self.window_size x self.window_size, E)
        x = x.unfold(1, W, W).unfold(2, W, W)
        x = x.contiguous().view(-1, W * W, E)

        # Patch-wise self-attention
        x, _ = self.attention(x, x, x)

        # Reshape back to original shape
        x = x.view(B, S // W, S // W, W, W, E)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, N, E)

        return x
    

class LocalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size, dim_feedforward=2048, dropout=0.05):
        super(LocalTransformerEncoderLayer, self).__init__()

        self.self_attn = Local2DAttention(d_model, nhead, window_size)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # self-attention
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out = F.gelu(x + out)
        return out


class Convolutionals(nn.Module):
    def __init__(self, n_layers, n_channels, hidden_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = n_channels
        for _ in range(n_layers - 1):
            self.layers.append(DoubleConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                residual=(in_channels == hidden_channels)
            ))
            in_channels = hidden_channels

        self.layers.append(DoubleConv(
            in_channels=hidden_channels,
            out_channels=n_channels,
            residual=(n_channels == hidden_channels)
        ))

    def forward(self, x):
        identity = x
        for layer in self.layers:
            out = layer(x)
            if x.shape == out.shape:
                out += identity
            x = out
            identity = out
        return out


class TBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, patch_size, num_classes=None, num_channels=3, dropout=0.05):
        super(TBlock, self).__init__()
        self.d_model = d_model
        self.patching_layer = PatchingLayer(patch_size, num_channels)
        self.projection = nn.Linear(num_channels * patch_size * patch_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dropout=dropout),
            num_layers
        )
        self.output_layer = OutputLayer(d_model, patch_size, num_channels)

        self.convs = Convolutionals(4, num_channels, 128) # EXPERIMENTAL

        if num_classes:
            self.label_emb = nn.Embedding(num_classes, d_model)

    def forward(self, x, t, label=None):
        # compute positional encoding for the timestep (len(t), self.time_dim)
        t = t.unsqueeze(-1).float()
        t = self._time_embedding(t)

        # class-conditioning
        if label:
            t = t + self.label_emb(label)

        x = self.patching_layer(x)
        x = self.projection(x)
        residual = x
        x = x + t + self.positional_encoding(x)
        x = self.encoder(x) + residual
        x = self.output_layer(x)
        x = x + self.convs(x) # EXPERIMENTAL
        return x

    def _time_embedding(self, t):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2, dtype=torch.float32).to(t.device) / self.d_model))

        # Create the sine and cosine encodings
        pos_enc_sin = torch.sin(t.unsqueeze(1).float() * inv_freq)
        pos_enc_cos = torch.cos(t.unsqueeze(1).float() * inv_freq)

        # Concatenate the sine and cosine encodings
        pos_enc = torch.cat([pos_enc_sin, pos_enc_cos], dim=-1)

        return pos_enc


class TNet(nn.Module):
    '''
    UNet but with a transformer instead of convolutions. Doesn't seem to work
    any better than the normal ImageTransformer
    '''
    def __init__(self, num_blocks, d_model, nhead, num_layers, patch_size, num_channels=3, dropout=0.05):
        super(TNet, self).__init__()

        # Encoder and Decoder blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks * 2):
            self.blocks.append(TBlock(d_model, nhead, num_layers, patch_size, num_channels=num_channels, dropout=dropout))

        self.pool = nn.AvgPool2d(2, 2)  # For downsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')  # For upsampling

    def forward(self, x, t, label=None):
        # Encoding path with downsampling
        encoder_outs = []
        for i in range(len(self.blocks) // 2):
            x = self.blocks[i](x, t, label)
            encoder_outs.append(x)
            x = self.pool(x)

        # Decoding path with upsampling
        for i in range(len(self.blocks) // 2, len(self.blocks)):
            x = self.upsample(x)
            x += encoder_outs.pop()  # Skip-connection
            x = self.blocks[i](x, t, label)
        
        return x

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)