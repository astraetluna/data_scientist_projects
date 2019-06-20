# Predict flower name from an image with predict.py along with the probability of that name. 
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

# Basic usage: python predict.py /path/to/image checkpoint
# Options:

# Imports here
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets, models
from torch import nn, optim
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plts
import argparse
import random, os
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image with predict.py along with the probability of that name")
    parser.add_argument("checkpoint", action="store", default="checkpoint.pth", help="Set checkpoint model file")
    parser.add_argument("--image_path", default=None, help="Set image path")
    # Return top K most likely classes: python predict.py input checkpoint --top_k 5
    parser.add_argument("--top_k", type=int, default =5, help="Set number of top categories")
    # Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    parser.add_argument("--category_names" , type=str, default="cat_to_name.json", help="Path to the category file")
    # Use GPU for inference: python predict.py input checkpoint --gpu
    parser.add_argument("--gpu", default=True, type=bool, help="Use GPU")
    args = parser.parse_args()
    return args

def loading_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = checkpoint['model']
    model.epochs = checkpoint['epochs']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    image = image.resize((256,256))
    image = image.crop((16,16,240,240))
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image.transpose(2,0,1)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
   
    #only if GPU is available
    if(gpu and torch.cuda.is_available()):
        image = process_image(image_path)
        image = torch.from_numpy(np.array([image])).float()
        image = Variable(image)
        image = image.cuda()
        output = model.forward(image)
        ps = torch.exp(output).data
        # probabilities
        prob = torch.topk(ps, topk)[0].tolist()[0] 
        #index 
        index = torch.topk(ps, topk)[1].tolist()[0] 
    
        ind = []
        for i in range(len(model.class_to_idx.items())):
            ind.append(list(model.class_to_idx.items())[i][0])

        # transfer index to label
        label = []
        for i in range(topk):
            label.append(ind[index[i]])
    else:
        print("Turn on GPU")
    return prob, label



def main():
    args = parse_args()
    gpu = args.gpu
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    img_path = args.image_path
    model= loading_checkpoint(args.checkpoint)
    prob, classes = predict(img_path, model, args.top_k, gpu)
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])

if __name__ == '__main__':
    main()