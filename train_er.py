"""
Script for training the ERROR RECOGNITION model
"""
from core.config import Config
import torch
from dataloader.CaptainCookSegmentDataset import CaptainCookSegmentDataset
from torch.utils.data import DataLoader
from test_er import test_er
from core.models.blocks import MLP
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


"""
Data type: ["Segments"]
Backbones:  [Slowfast, 3dresnet, x3d, Omnivore, Imagebind]
Modality: [Video]
Model: [ErMLP, ErCNN]

Output: 4 models

Data type: [Steps]
Backbones:  [Slowfast, 3dresnet, x3d, Omnivore]
Modality: [Video]
Model: [ErFormer]

Output: 5 models

Data type: [Steps]
Backbones: [Imagebind]
Modality: [Video, Audio, Text, Depth]
Model: [ErMMFormer]

Output: 4 models
"""


def train_er(config):
    # 1. Initialize the dataset
    # 2. Prepare the dataloader
    # 3. Initialize model based on the configuration
    # 4. Initialize the optimizer, learning rate scheduler, loss function
    # 5. In each epoch, pass the data through the model, calculate the loss, backpropagate the gradients, update the weights
    # 6. Save the model after each epoch - in /data/rohith/checkpoints/captaincook/er
    # 7. Evaluate the model after each epoch
    # 8. Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    args = config.args
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = CaptainCookSegmentDataset(config, 'train')
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_dataset = CaptainCookSegmentDataset(config, 'val')
    test_loader = DataLoader(test_dataset, **test_kwargs)

    for data, _ in train_loader:
        d_model = data.shape[1]

    model = MLP(d_model, d_model/3, 1).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch)
        test_er(model, device, test_loader, 'val')
        scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), f"{config.backbone}_{epoch}_MLP.pt")


def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader.dataset)} steps, Loss: {loss.item():.6f}')


if __name__ == "__main__":
    conf = Config()
    train_er(conf)
