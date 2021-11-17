""" This file contains the model Class used for the exps"""

import torch
import torch.nn as nn

from models.model import ViT
from models.transformer import Transformer

def remove_modulelist(network, all_layers):

    for layer in network.children():
        if isinstance(layer, Transformer) or isinstance(layer, nn.ModuleList): # if transformer/modulelist layer, apply recursively to layers in the layer
            #print(layer)
            remove_modulelist(layer, all_layers)
        else: # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)
            
"""def remove_DwsConvBlock(cur_layers):

    all_layers = []
    for layer in cur_layers:
        if isinstance(layer, DwsConvBlock):
           #  print("helloooo: ", layer)
            for ch in layer.children():
                all_layers.append(ch)
        else:
            all_layers.append(layer)
    return all_layers"""
            
class MyViT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        model_name = 'B_16_imagenet1k'
        model = ViT(model_name, pretrained=True, num_classes=50, image_size=128)
        self.class_token = nn.Parameter(torch.zeros(1, 1, model.dim))
        
        
            
        all_layers = []
        start_list = []
        lat_list = []
        end_list = []
        
        remove_modulelist(model, all_layers)
        
        for i, layer in enumerate(all_layers[:-1]):
            if i == 0:
                start_list.append(layer)
            elif i == 15:
                continue
            elif i <= 13:
                lat_list.append(layer)
            elif i > 14 and i < 14:
                continue
            else:
                end_list.append(layer)
        
        """for name, layer in model.named_children():
            if name == "patch_embedding":
                start_list.append(layer)
            elif name == "fc":
                continue
            else:
                lat_list.append(layer)"""
            #lat_list.append(layer)
        #remove_sequential(model, all_layers)
        #all_layers = remove_DwsConvBlock(all_layers)

        #lat_list = []
        #end_list = []

        """for i, layer in enumerate(all_layers[:-1]):
            if 
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)"""
        
        #self.all_features = nn.Sequential(*all_layers)
        self.start_features = nn.Sequential(*start_list)
        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        self.output = nn.Linear(768, 50, bias=False)
        #self.norm = nn.LayerNorm(768, eps=1e-6)
        
    def forward(self, x, latent_input=None, return_lat_acts=False):
    
        
        b, c, fh, fw = x.shape
        
        #input_tensor = x
        #print("X shape:", x.shape)
        """if latent_input is not None:
            #print("Latent_input shape", latent_input.shape)
            x = torch.cat((x, latent_input), 0)"""
        
        
        
        #print(self.start_features)
        #print(self.lat_features)
        #print(self.end_features)
    
        #print("Input tensor shape:", x.shape)
        
        embeddings = self.start_features(x)

        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = torch.cat((self.class_token.expand(b, -1, -1), embeddings), dim=1)
        
        #print("embeddings shape:", embeddings.shape)

        orig_acts = self.lat_features(embeddings)
        if latent_input is not None:
            lat_acts = torch.cat((orig_acts, latent_input), 0)
        else:
            lat_acts = orig_acts
        #lat_acts = orig_acts

        #print("lat_acts shape:", lat_acts.shape)
        
        #logits = self.end_features(lat_acts)
        #print("logits shape:", logits.shape)
        
        x = self.end_features(lat_acts)
        #x = x.view(x.size(0), -1)
        #print("x tensor shape:", x.shape)
        
        x = x[:, 0]
        #print("x tensor shape (after norm):", x.shape)
        logits = self.output(x)
        #print("logits tensor shape:", logits.shape)
        #logits = self.end_features(lat_acts)

        if return_lat_acts:
            return logits, orig_acts
            #return logits, input_tensor
        else:
            return logits
            

if __name__ == "__main__":

    model_name = 'B_16_imagenet1k'
    model = ViT(model_name, pretrained=True, num_classes=50)
    for name, param in model.named_parameters():
        print(name)
