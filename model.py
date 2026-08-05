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

        # Decoder: input channels doubled due to skip connection concatenation
        self.dec = nn.ModuleList([
            up_block(512, 256),  # no skip at bottleneck level
            up_block(256 + 256, 128),  # skip from enc[3]: 256ch
            up_block(128 + 128, 64),  # skip from enc[2]: 128ch
            up_block(64 + 64, 32),  # skip from enc[1]: 64ch
            up_block(32 + 32, 16),  # skip from enc[0]: 32ch
        ])
        self.out_conv = nn.Conv2d(16, 1, 1)

    def encode(self, x):
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
        return self.conv_mu(x), self.conv_logvar(x), skips

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, skips):
        x = self.conv_decode(z)
        for i, layer in enumerate(self.dec):
            if i > 0:
                x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = layer(x)
        return torch.sigmoid(self.out_conv(x))

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z, skips), mu, logvar
