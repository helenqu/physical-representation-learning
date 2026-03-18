import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from timm.models.layers import DropPath

class PatchEmbed3D(nn.Module):
    """
    patchify input 3D video, then embed
    """

    def __init__(
        self,
        patch_size=16,
        num_frames_per_patch=None,
        in_chans=2,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames_per_patch = num_frames_per_patch

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(num_frames_per_patch, patch_size, patch_size),
            stride=(num_frames_per_patch, patch_size, patch_size),
        ) # (B, C, T, H, W) -> (B, embed_dim, T//num_frames_per_patch, H//patch_size, W//patch_size)

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            weight_expanded = self.weight.view(self.weight.shape[0], *([1] * (x.dim() - 2)))
            bias_expanded = self.bias.view(self.bias.shape[0], *([1] * (x.dim() - 2)))
            x = weight_expanded * x + bias_expanded
            return x

class ResidualBlock(nn.Module):
    def __init__(self, embed_dim, num_spatial_dims=3, layer_scale_init_value=1e-6, drop_path=0.):
        super().__init__()
        self.num_spatial_dims = num_spatial_dims

        self.conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim) if num_spatial_dims == 3 else nn.Conv2d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim)
        self.norm = LayerNorm(embed_dim, data_format="channels_last")
        self.pwconv1 = nn.Linear(embed_dim, 4 * embed_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * embed_dim, embed_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((embed_dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1) if self.num_spatial_dims == 3 else x.permute(0, 2, 3, 1) # (N, C, T, H, W) -> (N, T, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) if self.num_spatial_dims == 3 else x.permute(0, 3, 1, 2) # (N, T, H, W, C) -> (N, C, T, H, W)

        x = input + self.drop_path(x)
        return x

class ConvEncoder(nn.Module):
    def __init__(self,
                 in_chans=2,
                 num_res_blocks=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 num_frames=4):
        super().__init__()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(1, 4, 4), padding='same'),
            LayerNorm(dims[0], data_format="channels_first"),
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)

        if num_frames == 16:
            for i in range(len(dims)-1):
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm(dims[i], data_format="channels_first"),
                        nn.Conv3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=2, stride=2)
                    )
                )
    
            self.res_blocks = nn.ModuleList()
            for i in range(len(dims)):
                self.res_blocks.append(
                    nn.Sequential(
                        *[ResidualBlock(dims[i], num_spatial_dims=3 if i < len(dims)-1 else 2) for _ in range(num_res_blocks[i])]
                    )
                )

        elif num_frames == 4:
            for i in range(3):
                stride = 2 if i % 2 == 0 else (1, 2, 2) # downsample time every other layer
                kernel_size = 2 if i % 2 == 0 else (1, 2, 2)
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm(dims[i], data_format="channels_first"),
                        nn.Conv3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, stride=stride)#, padding=(1, 0, 0)),
                    )
                )
            for i in range(3, len(dims)-1): # downsample spatial only
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm(dims[i], data_format="channels_first"),
                        nn.Conv2d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=2, stride=2)
                    )
                )
            
            self.res_blocks = nn.ModuleList()
            for i in range(len(dims)):
                self.res_blocks.append(
                    nn.Sequential(
                        *[ResidualBlock(dims[i], num_spatial_dims=3 if i < 3 else 2) for _ in range(num_res_blocks[i])]
                    )
                )

        else:
            raise ValueError(f"Currently supports 4 and 16 frames, input num_frames: {num_frames}")
        
        self.dims = dims

    def forward(self, x, **kwargs):
        for i in range(len(self.dims)):
            x = self.downsample_layers[i](x)
            x = x.squeeze(2)
            x = self.res_blocks[i](x)
        return x

class ConvEncoderViTTiny(nn.Module):
    def __init__(self,
                 in_chans=2,
                 num_res_blocks=[3, 3, 9, 3],
                 dims=[48, 96, 192, 384]):
        super().__init__()
        
        # Stem: 11 -> 48 channels, no spatial downsampling
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(1, 4, 4), padding='same'),
            LayerNorm(dims[0], data_format="channels_first"),
        )
        
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        
        # Layer 1: Time downsampling (4 -> 2), 48 -> 96 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[0], data_format="channels_first"),
                nn.Conv3d(in_channels=dims[0], out_channels=dims[1], 
                         kernel_size=(2, 1, 1), stride=(2, 1, 1)),  # downsample time only
            )
        )
        
         # Layer 2: Spatial downsampling (224 -> 112), 96 -> 192 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[1], data_format="channels_first"),
                nn.Conv2d(in_channels=dims[1], out_channels=dims[2], 
                         kernel_size=2, stride=2),  # downsample spatial
            )
        )
        
        # Layer 3: Spatial downsampling (112 -> 56), 192 -> 384 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[2], data_format="channels_first"),
                nn.Conv2d(in_channels=dims[2], out_channels=dims[3], 
                         kernel_size=2, stride=2),  # downsample spatial
            )
        )
        
        # Layer 4: Spatial downsampling (56 -> 28), keep 384 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[3], data_format="channels_first"),
                nn.Conv2d(in_channels=dims[3], out_channels=dims[3], 
                         kernel_size=2, stride=2),  # downsample spatial
            )
        )
        
        # Layer 5: Spatial downsampling (28 -> 14), keep 384 channels
        self.downsample_layers.append(
            nn.Sequential(
                LayerNorm(dims[3], data_format="channels_first"),
                nn.Conv2d(in_channels=dims[3], out_channels=dims[3], 
                         kernel_size=2, stride=2),  # downsample spatial
            )
        )
        
        # Residual blocks for each stage
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.downsample_layers)):
            if i == 0:
                channels = dims[0]
            elif i <= 3:
                channels = dims[i]
            else:
                channels = dims[3]  # 384 for final layers
            
            self.res_blocks.append(
                nn.Sequential(
                    *[ResidualBlock(channels, num_spatial_dims=3 if i <= 1 else 2) for _ in range(num_res_blocks[min(i, len(num_res_blocks)-1)])]
                )
            )
        
        self.dims = dims

    def forward(self, x, **kwargs):
        # Input: (B, 11, 4, 224, 224)
        # Layer 0: (B, 48, 4, 224, 224) - stem
        # Layer 1: (B, 96, 2, 224, 224) - time downsampling
        # Layer 2: (B, 192, 2, 112, 112) - spatial downsampling
        # Layer 3: (B, 384, 2, 56, 56) - spatial downsampling
        # Layer 4: (B, 384, 2, 28, 28) - spatial downsampling
        # Layer 5: (B, 384, 2, 14, 14) - spatial downsampling
        b, c0, t0, h0, w0 = x.shape
        for i in range(len(self.downsample_layers)):
            if i == 2:
                # flatten time dimension to use conv2ds
                b, c, t, h, w = x.shape
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
                x = x.view(b*t, c, h, w)  # (B*T, C, H, W)
            x = self.downsample_layers[i](x)
            x = self.res_blocks[i](x)
        # reshape back to (B, C, T, H, W)
        _, c, h, w = x.shape
        x = x.view(b, t0//2, c, h, w)  # (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        return x

class ConvPredictorViTTiny(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # self.thw = (time_dim, height_dim, width_dim)
        self.scale_factor = 2
        self.conv = nn.Sequential(
            nn.Conv3d(dims[0], dims[0]*self.scale_factor, kernel_size=2, padding=1),
            ResidualBlock(dims[0]*self.scale_factor, num_spatial_dims=3),
            nn.Conv3d(dims[0]*self.scale_factor, dims[0], kernel_size=2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvPredictor(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # self.thw = (time_dim, height_dim, width_dim)
        self.scale_factor = 2

        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[0]*self.scale_factor, kernel_size=2, padding=1),
            ResidualBlock(dims[0]*self.scale_factor, num_spatial_dims=2),
            nn.Conv2d(dims[0]*self.scale_factor, dims[0], kernel_size=2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvDecoder(nn.Module):
    # image input: 128x128x3 channels -> embedding: 4x4x768 channels -> input size
    def __init__(self,
                 out_chans=2,
                 num_res_blocks=[3, 9, 3, 3],
                 dims=[768, 384, 192, 96],
        ):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.upsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], data_format="channels_first"),
                    nn.ConvTranspose3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=2, stride=2),
                )
            )
        
        self.res_blocks = nn.ModuleList()
        for i in range(len(dims)-1):
            self.res_blocks.append(
                nn.Sequential(
                    *[ResidualBlock(dims[i]) for _ in range(num_res_blocks[i])]
                )
            )

        self.final_conv = nn.Conv3d(dims[-1], out_chans, kernel_size=1)
        
        self.dims = dims
    
    def forward(self, x):
        for i in range(len(self.dims)-1):
            x = self.res_blocks[i](x)
            x = self.upsample_layers[i](x)
        x = self.final_conv(x)
        return x

class RegressionHead(nn.Module):
    def __init__(self, in_dim, out_dim, flatten_first=False, add_dropout=False, dropout_rate=0.4):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.flatten_first = flatten_first
        self.add_dropout = add_dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.flatten_first:
            x = x.flatten(1, -1)
        if self.add_dropout:
            x = self.dropout(x)
        return self.fc(x)

class RegressionMLP(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim=32,
                 num_hidden_layers=1,
                 flatten_first=False,
                 add_dropout=False,
                 dropout_rate=0.2):
        super().__init__()

        self.add_dropout = add_dropout
        self.flatten_first = flatten_first
        
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp_block = lambda in_dim, out_dim: nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            self.dropout if add_dropout else nn.Identity(),
        )
        self.layers = []
        self.stem = self.mlp_block(in_dim, hidden_dim)
        self.hidden_layers = nn.Sequential(
            *[self.mlp_block(hidden_dim, hidden_dim) for _ in range(num_hidden_layers-1)]
        )
        self.output_layer = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        if self.flatten_first:
            x = x.flatten(1, -1)
        x = self.stem(x)
        if self.add_dropout:
            x = self.dropout(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)

class Projector3D(nn.Module):
    """
    Projector that takes embeddings of shape (B, C, T, H, W) and projects them to (B, C', T, H, W)
    using 1D Conv3D operations: C -> C' -> C'
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.proj = nn.Sequential(
            # First projection: C -> C'
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
            # Second projection: C' -> C'
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        Returns:
            Projected tensor of shape (B, C', T, H, W)
        """
        return self.proj(x)

# from convnext
def cosine_schedule_array(
    base_value,            # peak value after warmup (e.g., max LR)
    final_value,           # floor value at the very end (e.g., 1e-6)
    epochs=0,
    niter_per_ep=0,
    steps=0,
    warmup_epochs=0,
    start_warmup_value=0.0,
    warmup_steps=-1        # if >0, overrides warmup_epochs
):
    assert (epochs > 0 and niter_per_ep > 0) or steps > 0, "either (epochs and niter_per_ep) or steps must be provided"
    if steps == 0:
        total_steps = int(epochs * niter_per_ep)
        if total_steps <= 0:
            return np.array([], dtype=np.float32)
    else:
        total_steps = steps

    # Compute warmup iters (steps), prefer explicit steps if provided
    warmup_iters = int(warmup_steps) if warmup_steps > 0 else int(warmup_epochs * niter_per_ep)
    warmup_iters = max(0, min(warmup_iters, total_steps))

    # Warmup schedule (linear from start_warmup_value -> base_value)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters, dtype=np.float32
        )
    else:
        warmup_schedule = np.array([], dtype=np.float32)

    # Cosine decay schedule (base_value -> final_value)
    remain = total_steps - warmup_iters
    if remain > 0:
        # i ranges [0 .. remain-1]; use N-1 in denominator to hit final_value exactly at the last step
        if remain == 1:
            cos_factors = np.array([1.0], dtype=np.float32)  # single step = exactly base_value
        else:
            t = np.arange(remain, dtype=np.float32) / (remain - 1)
            # cosine from 0 -> pi
            cos_factors = 0.5 * (1.0 + np.cos(np.pi * t))
        decay_schedule = final_value + (base_value - final_value) * cos_factors
    else:
        decay_schedule = np.array([], dtype=np.float32)

    schedule = np.concatenate([warmup_schedule, decay_schedule]).astype(np.float32)
    # Safety clamp and length assert
    schedule = np.clip(schedule, min(start_warmup_value, final_value), max(base_value, final_value))
    assert len(schedule) == total_steps, f"len(schedule)={len(schedule)} != total_steps={total_steps}"
    return schedule

class CosineLRScheduler:
    def __init__(self, optimizer, step=0, **kwargs):
        self.optimizer = optimizer
        self.schedule = cosine_schedule_array(**kwargs)
        self.idx = step

    def step(self):
        if self.idx < len(self.schedule):
            lr = float(self.schedule[self.idx])
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            self.idx += 1
        # if idx >= schedule length, lr stays constant at final_value

    def get_last_lr(self):
        if self.idx == 0:
            return [pg["lr"] for pg in self.optimizer.param_groups]
        return [float(self.schedule[min(self.idx - 1, len(self.schedule) - 1)])]

    def state_dict(self):
        return {"idx": self.idx, "schedule": self.schedule.tolist()}

    def load_state_dict(self, state_dict):
        self.idx = state_dict["idx"]
        self.schedule = np.array(state_dict["schedule"], dtype=np.float32)