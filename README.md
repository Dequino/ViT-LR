# ViT-LR
A latent replay model using Vision Transformer (ViT) as backbone

## Usage
Requires the AR1 conda enviroment found [here](https://github.com/vlomonaco/ar1-pytorch) and the Core50 database found [here](https://github.com/vlomonaco/core50).

Edit the Code50 root folder path in the source code when creating the database object.

'''
python3 test.py
'''

Prints the model structure and parameters.

'''
python3 vitlr.py
'''

Trains the model on the Core50 dataset, nicv2_391 scenario. A tensorboard logs folder and a state dict will be generated.

'''
python3 vitlr_stats.py
'''

Prints the average training speed of single image batch

Jupyter notebooks:
1. Confusion matrix
Prints the confusion matrix of a trained model, requires the output state dict generated by vitlr.py
2. Attention rollout
Prints a visual representation of the attention rollout, mostly experimental, requires a state dict.

## How to edit the model
Coming soon


