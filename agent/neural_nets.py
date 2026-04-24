import torch.nn as nn
import torch
import numpy as np


class Policy(nn.Module):
    def __init__(
        self,
        num_classes,
        device: str | None = None,
        **kwargs
    ):
        super().__init__()
        self.state_encoder = ResNet18(in_channels=5, **kwargs)
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(device)

    def forward(self, obs):
        # Move inputs to device
        obs = obs.to(self.device)
        state_features = self.state_encoder(obs)
        x = self.classification_head(state_features)  # (N, 512) -> (N, num_classes)
        return x
    
class Critic(nn.Module):
    def __init__(
        self,
        device: str | None = None,
        **kwargs
    ):
        super().__init__()
        self.state_encoder = ResNet18(in_channels=5, **kwargs)
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(device)

    def forward(self, obs):
        # Move inputs to device
        obs = obs.to(self.device)
        state_features = self.state_encoder(obs)
        x = self.regression_head(state_features)  # (N, 512) -> (N, num_classes)
        return x

class StateEncoder(nn.Module):
    def __init__(
        self,
        grid_channels: int = 5,
        num_orientations: int = 4,
        orientation_embedding_dim: int = 64,
    ):
        super().__init__()
        self.grid_encoder = ResNet18(in_channels=grid_channels)  # (N, C, H, W) -> (N, 512)
        self.orientation_embedding = nn.Embedding(
            num_embeddings=num_orientations,
            embedding_dim=orientation_embedding_dim,
        )
        resnet_out_channels = 512
        self.fusion_module = FusionModule(
            grid_encoding_dim=resnet_out_channels,
            orientation_embedding_dim=orientation_embedding_dim,
        )

    def forward(self, grid, orientation):
        grid_encoding = self.grid_encoder(grid)  # (N, C, H, W) -> (N, 512)
        orientation_embedding = self.orientation_embedding(orientation)
        final_features = self.fusion_module(grid_encoding, orientation_embedding)
        return final_features


class FusionModule(nn.Module):
    def __init__(self, grid_encoding_dim: int = 512, orientation_embedding_dim: int = 64):
        super().__init__()
        in_features = grid_encoding_dim + orientation_embedding_dim
        self.fusion_layers = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU()
        )

    def forward(self, grid_encoding, orientation_embedding):
        x = torch.concat([grid_encoding, orientation_embedding], dim=-1)
        y = self.fusion_layers(x)
        return y

class ResNet8(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Set network hyperparameters
        self.block_kernel_size = [3, 3, 3]
        self.block_in_channels = [64, 128, 256]
        self.block_out_channels = [128, 256, 512]
        self.block_stride = [2, 2, 2]
        # Initialize network
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        block_params = zip(
            self.block_in_channels,
            self.block_out_channels,
            self.block_kernel_size,
            self.block_stride,
        )
        for in_channels, out_channels, kernel_size, stride in block_params:
            self.network.append(ResNetBlock(in_channels, out_channels, kernel_size, stride))
        self.network.append(nn.AdaptiveAvgPool2d(output_size=1))  # (N, 512, 3, 4) -> (N, 512, 1, 1)
        self.network.append(nn.Flatten())  # (N, 512)

    def forward(self, x):
        y = self.network(x)  # (N, 512)
        return y


class ResNet18(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Set network hyperparameters
        self.block_kernel_size = [3, 3, 3, 3, 3, 3, 3, 3]
        self.block_in_channels = [64, 64, 64, 128, 128, 256, 256, 512]
        self.block_out_channels = [64, 64, 128, 128, 256, 256, 512, 512]
        self.block_stride = [1, 1, 2, 1, 2, 1, 2, 1]
        # Initialize network
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        block_params = zip(
            self.block_in_channels,
            self.block_out_channels,
            self.block_kernel_size,
            self.block_stride,
        )
        for in_channels, out_channels, kernel_size, stride in block_params:
            self.network.append(ResNetBlock(in_channels, out_channels, kernel_size, stride))
        self.network.append(nn.AdaptiveAvgPool2d(output_size=1))  # (N, 512, 3, 4) -> (N, 512, 1, 1)
        self.network.append(nn.Flatten())  # (N, 512)

    def forward(self, x):
        y = self.network(x)  # (N, 512)
        return y


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
        )
        if in_channels != out_channels:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )
        else:
            self.projection = None
        self.activation = nn.PReLU()

    def forward(self, x):
        z = self.layers(x)  # (N, in_C, H, W) -> (N, out_C, H / stride, W / stride)
        if self.projection:
            x = self.projection(x)  # (N, in_C, H, W) -> (N, out_C, H / stride, W / stride)
        z = z + x  # (N, out_C, H / stride, W / stride)
        y = self.activation(z)
        return y
