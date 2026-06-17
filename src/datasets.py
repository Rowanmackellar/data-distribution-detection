# Both clean and shifted data sets, as well as methods achieving such.
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import platform

# Automatically drops to 0 on Mac to completely avoid pickling/process spawning issues
NUM_WORKERS = 0 if platform.system() == "Darwin" else 2

class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=2.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
    def __call__(self, img):
        import torchvision.transforms.functional as F
        return F.gaussian_blur(img, [self.kernel_size, self.kernel_size], [self.sigma, self.sigma])

# NEW: Replaced the lambda function with a picklable class helper
class AddGaussianNoise:
    def __init__(self, std=0.15):
        self.std = std
    def __call__(self, tensor):
        return tensor + self.std * torch.randn_like(tensor)


def get_clean_dataloaders(data_dir="data/raw", batch_size=128):
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), norm
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), norm])
    
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    
    # Swapped hardcoded '2' with NUM_WORKERS
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS),
            DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS))


def get_shifted_dataloader(shift_type="blur", data_dir="data/raw", batch_size=128):
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    if shift_type == "blur":
        tfs = transforms.Compose([transforms.ToTensor(), GaussianBlur(), norm])
    elif shift_type == "noise":
        # Using the clean AddGaussianNoise helper instead of the lambda
        tfs = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(), norm])
    else:
        raise ValueError(f"Unknown shift: {shift_type}")
        
    shifted_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfs)
    
    # Swapped hardcoded '2' with NUM_WORKERS
    return DataLoader(shifted_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)