import os
import torch
import argparse
import logging
from multiprocessing import set_start_method
from src.train_modes import pretrain
from src.datasets import PretrainDatasetOSVMini
from transformers import Trainer, TrainingArguments, \
                         AutoModelForImageClassification, \
                         CLIPVisionModel, CLIPModel, CLIPProcessor

logger = logging.getLogger('run')

def unfreeze_all_but_last(model):
    """Unfreezes the parameters of a model.
    for p in model.parameters():
        p.requires_grad = True"""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    model_parameters = model.named_parameters()
    for name, param in model_parameters:
        if len(name.split(".")) > 5:
            if name.split(".")[4] == "11":
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
    model.train() # assume this is wanted...

    
def main():
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name)
    unfreeze_all_but_last(model.vision_model)
    
    # Verify the freezing worked
    # print(model.vision_model.encoder.layers)
    # print(model.vision_model.encoder.layers[11].layer_norm1.weight.requires_grad)
    # print(model.vision_model.encoder.layers[11].layer_norm1.bias.requires_grad)
    # print(model.vision_model.encoder.layers[11].layer_norm2.weight.requires_grad)
    # print(model.vision_model.encoder.layers[11].layer_norm2.bias.requires_grad)
    print(model.base_model)
    
if __name__ == '__main__':
    main()