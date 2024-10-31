# Error Recognition

Please download or extract features for the data following the instructions in the [Feature Extraction](https://github.com/CaptainCook4D/feature_extractors) repository.

Once you have the features, you can run the code in this repository to train the models.

We have code three tasks: 

1. Error Recognition
   - Multimodal Data 
     - Video
     - Audio
     - Text
     - Depth
2. Early Error Recognition
   - Video Data


## How to run the code

1. Download or extract features for the data. 
2. Place the features in the respective folders.
3. Change the paths in the /core/config.py file.
4. Copy the respective script from scripts folder to the root directory.
5. Run the script.

## How to evaluate the models

Our reported results correspond to the following threshold values:

1. Step Split: 0.6
2. Recordings Split: 0.4

Note the best epoch of the trained models and add it accordingly to the /core/evaluate.py file.

Use it to get the evaluation csvs for different thresholds.