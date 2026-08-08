import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def down_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.enc = nn.ModuleList([
            down_block(1, 32),
            down_block(32, 64),
            down_block(64, 128),
            down_block(128, 256),
            down_block(256, 512),
        ])

        # Bottleneck
        self.conv_mu = nn.Conv2d(512, latent_dim, 1)
        self.conv_logvar = nn.Conv2d(512, latent_dim, 1)
        self.conv_decode = nn.Conv2d(latent_dim, 512, 1)

        # Decoder
        self.dec = nn.ModuleList([
            up_block(512, 256),
            up_block(256, 128),
            up_block(128, 64),
            up_block(64, 32),
            up_block(32, 16),
        ])
        self.out_conv = nn.Conv2d(16, 1, 1)

    def encode(self, x):
        for layer in self.enc:
            x = layer(x)
        return self.conv_mu(x), self.conv_logvar(x)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        x = self.conv_decode(z)
        for layer in self.dec:
            x = layer(x)
        return torch.sigmoid(self.out_conv(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar
