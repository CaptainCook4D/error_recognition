import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from torch import nn
from tqdm import tqdm


def test_er(model, device, test_loader, phase, criterion):
    model.eval()
    test_loss = 0
    total_samples = 0
    all_targets = []
    all_outputs = []

    test_loader = tqdm(test_loader)
    num_batches = len(test_loader)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_samples += data.shape[0]
            test_loss += criterion(output, target).sum().item()
            all_outputs.append(output.sigmoid().detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))

            # Set the description of the tqdm instance to show the loss
            test_loader.set_description(f'{phase} Progress: {total_samples}/{num_batches}')

    # Flatten lists
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Calculate metrics
    test_loss /= total_samples
    pred_labels = (all_outputs > 0.5).astype(int)
    precision = precision_score(all_targets, pred_labels)
    recall = recall_score(all_targets, pred_labels)
    f1 = f1_score(all_targets, pred_labels)
    auc = roc_auc_score(all_targets, all_outputs)

    # Print results
    print(
        f'\n{phase} set: Average loss: {test_loss:.4f}, Accuracy: {np.sum(pred_labels == all_targets)}/{pred_labels.shape[0]} '
        f'({100. * np.mean(pred_labels == all_targets):.0f}%)')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC: {auc:.4f}\n')