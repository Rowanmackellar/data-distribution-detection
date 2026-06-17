import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from train import get_device
from src.models import ResNetCIFAR
from src.datasets import get_shifted_dataloader

# CIFAR-10 class labels for human-readable plotting
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@torch.no_grad()
def find_confident_failures(model, loader, device, num_samples=5):
    model.eval()
    
    confident_failures = []
    
    for imgs, labels in loader:
        imgs_dev = imgs.to(device)
        outputs = model(imgs_dev)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        
        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        targets = labels.numpy()
        
        # Look for instances where prediction is wrong, but confidence is high
        for i in range(len(targets)):
            if preds[i] != targets[i]:
                confident_failures.append({
                    'image': imgs[i], # Keep original tensor for un-normalizing/plotting
                    'target': targets[i],
                    'pred': preds[i],
                    'conf': confidences[i]
                })
                
    # Sort failures by confidence in descending order (biggest blunders first)
    confident_failures.sort(key=lambda x: x['conf'], reverse=True)
    return confident_failures[:num_samples]

def plot_failures(failures, save_path="plots/confident_failures.png"):
    """Plots a row of high-confidence mistakes."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_plots = len(failures)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 4))
    if num_plots == 1:
        axes = [axes]
        
    for i, item in enumerate(failures):
        # Denormalize image for visualization
        img = item['image'].numpy().transpose((1, 2, 0))
        # Simple clipping approximation to bring image back to [0,1] range for viewing
        img = np.clip(img * 0.25 + 0.5, 0, 1) 
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        title = (f"True: {CLASSES[item['target']]}\n"
                 f"Pred: {CLASSES[item['pred']]}\n"
                 f"Conf: {item['conf']:.2%}")
        axes[i].set_title(title, fontsize=10, color='red')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Successfully saved confident failure visualization to {save_path}")

if __name__ == "__main__":
    device = get_device()
    
    # 1. Load the model
    model = ResNetCIFAR().to(device)
    model.load_state_dict(torch.load("models/checkpoints/resnet18_best.pt", map_location=device))
    
    # 2. Grab a shifted dataloader (let's use blur)
    print("Loading blurry shifted data...")
    blur_loader = get_shifted_dataloader(shift_type="blur")
    
    # 3. Extract the top 5 confident errors
    print("Analyzing model blunders...")
    top_failures = find_confident_failures(model, blur_loader, device, num_samples=5)
    
    # 4. Generate the visualization
    if top_failures:
        plot_failures(top_failures)
    else:
        print("Wow! No failures found.")