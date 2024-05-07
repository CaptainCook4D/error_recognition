"""
Script for training the ERROR RECOGNITION model
"""
from core.config import Config

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


def train(config):
    # 1. Initialize the dataset
    # 2. Prepare the dataloader
    # 3. Initialize model based on the configuration
    # 4. Initialize the optimizer, learning rate scheduler, loss function
    # 5. In each epoch, pass the data through the model, calculate the loss, backpropagate the gradients, update the weights
    # 6. Save the model after each epoch - in /data/rohith/checkpoints/captaincook/er
    # 7. Evaluate the model after each epoch
    # 8. Save the evaluation metrics in /logs/ as csv file and also push it to firebase database
    pass


if __name__ == "__main__":
    conf = Config()
    train(conf)
