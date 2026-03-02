import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import random
import matplotlib.pyplot as plt
import cv2
import coremltools as ct
from model import MiniUNet

# parameters
IMG_W, IMG_H = 192, 256       # input image size
OUT_W, OUT_H = 192, 256       # depth map size
BATCH_SIZE = 6
EPOCHS = 120
LR = 1e-4

# load data
X = np.load("image_frames.npy").astype(np.float32) / 255.0
Y = np.load("dists.npy").astype(np.float32)
Y = np.clip(Y / 5.0, 0.0, 1.0)

# move frame axis to front
X = np.moveaxis(X, -1, 0)
Y = np.moveaxis(Y, -1, 0)

# add channel dim
X = X[:, None, :, :]
Y = Y[:, None, :, :]

# split train/val
indices = list(range(len(X)))
random.shuffle(indices)
split = int(len(indices) * 0.86)

train_idx, val_idx = indices[:split], indices[split:]
train_X, val_X = X[train_idx], X[val_idx]
train_Y, val_Y = Y[train_idx], Y[val_idx]


np.save("image_frames_validation.npy", val_X.astype(np.float32))
np.save("dists_validation.npy", val_Y.astype(np.float32))

print("Saved validation sets:")
print(" - image_frames_validation.npy")
print(" - dists_validation.npy")


# dataset class
class DepthDataset(data.Dataset):
    def __init__(self, imgs, maps):
        self.imgs = imgs
        self.maps = maps

        # precompute the list of random scales 0.99 - 0.70
        self.scales = np.arange(0.80, 1.00, 0.01)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx][0]  # (960,720) numpy
        depth = self.maps[idx]   # (1,256,192) torch target

        # pick a random scale
        s = random.choice(self.scales)
        ds_w = int(IMG_W * s)
        ds_h = int(IMG_H * s)

        # downscale & upscale again
        img_aug = cv2.resize(img, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        img_aug = cv2.resize(img_aug, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        # restore channel dim
        img_aug = img_aug[None, :, :]

        x = torch.tensor(img_aug, dtype=torch.float32)
        y = torch.tensor(depth, dtype=torch.float32)

        return x, y


train_loader = data.DataLoader(DepthDataset(train_X, train_Y), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = data.DataLoader(DepthDataset(val_X, val_Y), batch_size=BATCH_SIZE)






# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = MiniUNet().to(device)
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()


# training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        opt.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            val_loss += loss_fn(y_pred, y).item()

    val_loss /= len(val_loader)


    print(f"epoch {epoch+1}/{EPOCHS} - train {train_loss:.4f}  val {val_loss:.4f}")

# save trained model
torch.save(model.state_dict(), "depth_model.pth")
print("\nmodel saved as depth_model.pth")

# --- re-create on CPU for tracing / CoreML ---
cpu_model = MiniUNet()
cpu_model.load_state_dict(torch.load("depth_model.pth", map_location="cpu"))
cpu_model.eval()

dummy = torch.randn(1, 1, 256, 192, dtype=torch.float32)

with torch.no_grad():
    traced = torch.jit.trace(cpu_model, dummy)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=dummy.shape)]
)
mlmodel.save("DepthModel.mlpackage")
print("Saved DepthModel.mlpackage")
