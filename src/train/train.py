import torch
import torch.nn as nn
import torch.optim as optim
from src.core.device import torch_device
from src.core.models import SimpleCNN
from src.core.datasets import get_fer2013_loaders

def train_model():
    device = torch.device(torch_device())
    train_loader, val_loader = get_fer2013_loaders()

    model = SimpleCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):  #  5 epoch as demo
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Val Acc: {100*correct/total:.2f}%")

    torch.save(model.state_dict(), "models/fer2013_cnn.pth")

if __name__ == "__main__":
    train_model()

