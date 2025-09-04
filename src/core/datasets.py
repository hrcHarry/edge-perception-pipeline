import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_fer2013_loaders(data_dir="data/fer2013", batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "test")   # FER2013 沒有 validation，就用 test 當 val

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds   = datasets.ImageFolder(val_dir,   transform=transform)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ld, val_ld
