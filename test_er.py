"""
Script for testing the ERROR RECOGNITION model
"""
from dataloader.CaptainCookSegmentDataset import CaptainCookSegmentDataset
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def test_er(model, device, test_loader, phase):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_samples += data.shape[0]
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = (output > 0.5).int()
            correct += pred.eq(target).sum().item()

    test_loss /= total_samples

    print(f'\n{phase} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({100. * correct / total_samples:.0f}%)\n')