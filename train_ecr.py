"""
Script for training the ERROR CATEGORY RECOGNITION model
"""

"""
Data type: [Segments]
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