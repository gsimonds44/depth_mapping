import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ======================
        #      ENCODER
        # ======================

        # Level 1: 256x192 -> 128x96
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        # Level 2: 128x96 -> 64x48
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Level 3: 64x48 -> 32x24
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # Level 4: 32x24 -> 16x12
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 192, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2)

        # ======================
        #      BOTTLENECK
        # ======================
        self.mid = nn.Sequential(
            nn.Conv2d(192, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)
        )

        # ======================
        #      DECODER
        # ======================

        # Up from mid -> level 4 (16x12 -> 32x24)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 192, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(192 + 192, 192, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Up from level 4 -> level 3 (32x24 -> 64x48)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Up from level 3 -> level 2 (64x48 -> 128x96)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Up from level 2 -> level 1 (128x96 -> 256x192)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Output layer
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # ----- Encoder -----
        x1 = self.conv1(x)
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        x4 = self.conv4(p3)
        p4 = self.pool4(x4)

        # ----- Bottleneck -----
        m = self.mid(p4)

        # ----- Decoder -----
        u4 = self.up4(m)
        u4 = torch.cat([u4, x4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)

        return torch.sigmoid(self.out(d1))
