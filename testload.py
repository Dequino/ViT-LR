from data_loader import CORE50
import copy
import os
import json
from models.vit import MyViT
from utils import *
import configparser
import argparse
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

dataset = CORE50(root='/home/adequino/work/ar1-pytorch/core50', scenario="nicv2_391")
preproc = preprocess_imgs

test_x, test_y = dataset.get_test_set()

model = MyViT(pretrained=True)

model.load_state_dict(torch.load("state_dict_model.pt"))

model.eval()
criterion = torch.nn.CrossEntropyLoss()
mb_size = 10

ave_loss, acc, accs = get_accuracy(
        model, criterion, mb_size, test_x, test_y, preproc=preproc
    )

print("---------------------------------")
print("Accuracy: ", acc)
print("---------------------------------")