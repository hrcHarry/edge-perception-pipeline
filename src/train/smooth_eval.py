# src/train/smooth_eval.py
import os, csv
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.core.models import SimpleCNN
from src.robustness.kalman import kf_smooth_probs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=7).to(device)
state = torch.load("models/fer2013_cnn.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

tfm = transforms.Compose([transforms.Grayscale(1), transforms.Resize((48,48)), transforms.ToTensor()])
val_ds = datasets.ImageFolder("data/fer2013/test", transform=tfm)
# val_ds = datasets.FakeData(transform=tfm, size=400)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

logits_list, ys = [], []
with torch.no_grad():
    for x, y in val_ld:
        x = x.to(device)
        logits = model(x)          # (B, C)
        logits_list.append(logits.cpu())
        ys.append(y)
logits_all = torch.cat(logits_list, dim=0)  # (T, C)
y_all = torch.cat(ys, dim=0)                # (T,)

probs = F.softmax(logits_all, dim=1)                # (T, C)
pred_raw = probs.argmax(dim=1)
acc_raw = (pred_raw == y_all).float().mean().item()

probs_s = kf_smooth_probs(probs, q=1e-3, r=1e-2)    # Kalman smoothing
pred_s = probs_s.argmax(dim=1)
acc_s = (pred_s == y_all).float().mean().item()

print(f"raw acc   = {acc_raw:.4f}")
print(f"kalman acc= {acc_s:.4f}")

os.makedirs("results", exist_ok=True)
with open("results/smooth_eval.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "value"])
    w.writerow(["acc_raw", acc_raw]); w.writerow(["acc_kalman", acc_s])

cls = 0
T = min(200, probs.size(0))
plt.figure()
plt.plot(probs[:T, cls].numpy(), label="raw")
plt.plot(probs_s[:T, cls].numpy(), label="kalman")
plt.title(f"Class-{cls} prob (first {T} steps)")
plt.xlabel("t"); plt.ylabel("prob")
plt.legend(); plt.tight_layout()
plt.savefig("results/smooth_trace.png")

print("Saved results/smooth_eval.csv and results/smooth_trace.png")

