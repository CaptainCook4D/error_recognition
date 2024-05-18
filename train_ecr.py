import torch.nn.functional as F
import core.models.blocks as blocks

"""
Script for training the ERROR CATEGORY RECOGNITION model
"""

"""
Data type: [Segments]
Backbones:  [Slowfast, 3dresnet, x3d, Omnivore, Imagebind]
Modality: [Video]
Model: [ErMLP, ErCNN]

Output: 5 models

Data type: [Steps]
Backbones:  [Slowfast, 3dresnet, x3d, Omnivore]
Modality: [Video]
Model: [ErFormer]

Output: 4 models

Data type: [Steps]
Backbones: [Imagebind]
Modality: [Video, Audio, Text, Depth]
Model: [ErMMFormer]

Output: 4 models
"""


def train_ecr(args, model, optimizer, device, data_loader, epoch):
    backbone = args.backbone
    modality = args.modality
    phase = args.phase
    segment_length = args.segment_length

    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        # load batch of data into GPU
        data, target = data.to(device), data.to(device)
        # set gradients of all parameters to zero
        optimizer.zero_grad()
        # calculate output from model for the batch
        output = model(data)
        # calculate loss for the batch
        loss = F.cross_entropy(output, target)
        # backpropogate the loss
        loss.backward()
        # update parameters of the model
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/ {len(data_loader.dataset)} ' f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break


if __name__ == "__main__":
    pass
