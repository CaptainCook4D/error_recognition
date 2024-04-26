import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import wandb


def train(model, dataloader, optimizer, epoch, num_epochs):
    criterion = nn.BCEWithLogitsLoss()
    model.train()  # Set the model to training mode
    running_loss = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Iterate over data
    for batch in tqdm(
        dataloader, desc=f"Epoch {epoch}/{num_epochs}", total=len(dataloader)
    ):
        inputs, labels = (
            batch[0],
            batch[1],
        )  # Considering only features and labels from the batch

        # Move data to GPU if available
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    wandb.log({"train_loss": epoch_loss})
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    return epoch_loss


@torch.no_grad()
def test(model, dataloader, device):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()  # Set the model to evaluation mode
    all_outputs = []
    all_labels = []
    all_indices_for_split = []
    all_recording_ids = []
    running_loss = 0.0

    for batch_idx, (data) in enumerate(tqdm(dataloader)):
        features, labels, _, _, index_for_split, recording_id = data

        # Move data to GPU if available
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        loss = criterion(outputs.view(-1), labels)
        running_loss += loss.item() * features.size(0)
        # Convert to list and extend
        # input_tensors = features.detach().cpu().numpy().tolist()
        output_tensors = outputs.detach().cpu().numpy().tolist()
        index_for_split = index_for_split
        recording_id = recording_id
        labels = labels.detach().cpu().numpy().tolist()
        all_outputs.extend(output_tensors)
        all_labels.extend(labels)
        all_indices_for_split.extend(index_for_split)
        all_recording_ids.extend(recording_id)
    epoch_loss = running_loss / len(dataloader.dataset)
    wandb.log({"test_loss": epoch_loss})
    print(f"Test Loss: {epoch_loss:.4f}")
    return (
        epoch_loss,
        np.array(all_outputs),
        np.array(all_labels),
        np.array(all_indices_for_split),
        np.array(all_recording_ids),
    )

    # Accumulating data for each step (index_for_split acts as step here)
