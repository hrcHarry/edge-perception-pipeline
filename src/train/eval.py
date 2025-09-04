import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from src.core.models import SimpleCNN
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=7).to(device)
state = torch.load("models/fer2013_cnn.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()

tfm = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# val_ds = datasets.FakeData(transform=tfm, size=200)
val_ds = datasets.ImageFolder("data/fer2013/test", transform=tfm)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

correct, total = 0, 0
losses = []

with torch.no_grad():
    for x, y in val_ld:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

acc = correct / total
print(f"Validation accuracy: {acc:.4f}, avg loss: {sum(losses)/len(losses):.4f}")

plt.plot(losses)
plt.title("Validation Loss per Batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
os.makedirs("results", exist_ok=True)
plt.savefig("results/eval_loss_curve.png")
print("Saved results/eval_loss_curve.png")

