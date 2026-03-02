import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import MiniUNet


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniUNet().to(device)
model.load_state_dict(torch.load("depth_model.pth", map_location=device))
model.eval()

print("Loaded depth_model.pth")

# load data
X = np.load("image_frames_validation.npy").astype(np.float32)
Y = np.load("dists_validation.npy").astype(np.float32)
N = len(X)

# compute per-pixel error histogram
print("\nRunning full-dataset inference to compute error histogram...\n")

all_errors_cm = []

model.eval()
with torch.no_grad():
    for i in range(N):
        # ground truth depth (0-1 normalized)
        gt = Y[i, 0]  # (H, W)

        # inference
        t = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
        pred = model(t).cpu().numpy()[0, 0]

        # normalized absolute error (0-1)
        err = np.abs(pred - gt)

        # convert to cm (1.0 = 500 cm)
        err_cm = err * 500.0

        # flatten
        all_errors_cm.append(err_cm.reshape(-1))

# merge all
all_errors_cm = np.concatenate(all_errors_cm)
print(f"Collected {len(all_errors_cm):,} pixel errors")

print("Mean error (cm):", np.mean(all_errors_cm))
print("Median error (cm):", np.median(all_errors_cm))
print("90th percentile (cm):", np.percentile(all_errors_cm, 90))


# plot histogram in cm with percentile markers
plt.figure(figsize=(10,4))

# plot histogram
counts, bins, _ = plt.hist(
    all_errors_cm,
    bins=2000,
    color='royalblue',
    alpha=0.85
)

# compute percentiles
p50 = np.percentile(all_errors_cm, 50)   # median
p70 = np.percentile(all_errors_cm, 70)
p90 = np.percentile(all_errors_cm, 90)

# vertical percentile lines
plt.axvline(p50, color='red', linewidth=0.5, label=f"50th Percentile: {p50:.2f} cm")
plt.axvline(p70, color='orange', linewidth=0.5, label=f"70th Percentile: {p70:.2f} cm")
plt.axvline(p90, color='green', linewidth=0.5, label=f"90th Percentile: {p90:.2f} cm")

plt.title("Pixel-wise Depth Error (cm)")
plt.xlabel("Absolute Error (centimeters)")
plt.ylabel("Pixel Count")
plt.grid(alpha=0.3)
plt.legend()

# limit x axis to 0-200 cm
plt.xlim(0, 100)

plt.tight_layout()
plt.show()




# arrow-key viewer
current = 0

fig = plt.figure(figsize=(10, 4))

def update():
    global current
    plt.clf()

    img = X[current,0]
    target = Y[current,0]

    t = torch.tensor(X[current:current+1], dtype=torch.float32).to(device)
    pred = model(t).detach().cpu().numpy()[0, 0]

    error = np.abs(pred - target)

    # panels
    ax1 = fig.add_subplot(1,4,1)
    ax1.set_title(f"Input {current}")
    ax1.imshow(img, cmap="gray"); ax1.axis("off")

    ax2 = fig.add_subplot(1,4,2)
    ax2.set_title("Truth")
    ax2.imshow(target, cmap="viridis"); ax2.axis("off")

    ax3 = fig.add_subplot(1,4,3)
    ax3.set_title("Prediction")
    ax3.imshow(pred, cmap="viridis"); ax3.axis("off")

    ax4 = fig.add_subplot(1,4,4)
    ax4.set_title("Error Magnitude")
    ax4.imshow(error, cmap="inferno")
    ax4.axis("off")

    plt.tight_layout()
    plt.draw()

def on_key(event):
    global current
    if event.key == "right":
        current = (current + 1) % N
    elif event.key == "left":
        current = (current - 1) % N
    update()

fig.canvas.mpl_connect("key_press_event", on_key)
update()
plt.show()
