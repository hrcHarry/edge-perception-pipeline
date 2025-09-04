import os, csv
import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.core.models import SimpleCNN
from src.robustness import noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading model
model = SimpleCNN(num_classes=7).to(device)
state = torch.load("models/fer2013_cnn.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

# dataset
tfm_base = transforms.Compose([transforms.Grayscale(1), transforms.Resize((48, 48))])
to_tensor = transforms.ToTensor()

val_ds = datasets.ImageFolder("data/fer2013/test", transform=tfm_base)
# val_ds = datasets.FakeData(transform=tfm_base, size=400)

def acc_eval(ds, perturb=None):
    correct = total = 0
    with torch.no_grad():
        for x, y in ds:
            img = transforms.ToPILImage()(x) if isinstance(x, torch.Tensor) else x
            if perturb is not None:
                img = perturb(img)
            x_t = to_tensor(img).unsqueeze(0).to(device)
            logits = model(x_t)
            pred = logits.argmax(1).item()
            correct += int(pred == y)
            total += 1
    return correct / max(1, total)

scenarios = [
    ("clean",        None),
    ("gaussian_0.10", lambda im: noise.gaussian(im, std=0.10)),
    ("blur_k5",      lambda im: noise.motion_blur(im, k=5)),
    ("occlusion_25", lambda im: noise.occlusion(im, frac=0.25)),
    ("bright_1.4",   lambda im: noise.brightness(im, factor=1.4)),
]

results = []
for name, fn in scenarios:
    a = acc_eval(val_ds, fn)
    print(f"{name:12s} acc = {a:.4f}")
    results.append((name, a))

os.makedirs("results", exist_ok=True)
with open("results/robust_eval.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "accuracy"])
    w.writerows(results)

plt.figure()
plt.bar([r[0] for r in results], [r[1] for r in results])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Accuracy")
plt.title("Clean vs Noisy Accuracy")
plt.tight_layout()
plt.savefig("results/robust_bar.png")
print("Saved results/robust_eval.csv and results/robust_bar.png")
