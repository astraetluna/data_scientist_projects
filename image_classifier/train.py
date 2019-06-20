# Train a new network on a data set with train.py
# Prints out training loss, validation loss, and validation accuracy as the network trains


# Imports here
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets, models
from torch import nn, optim
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint")
    # Basic usage: python train.py data_directory
    parser.add_argument("--data_dir", default="flowers", type=str, help="Choose your data path")
    # Choose architecture: python train.py data_dir --arch "vgg13"
    parser.add_argument("--arch", default="vgg16", type=str, help="Choose your model architecture")
    # Use GPU for training: python train.py data_dir --gpu
    parser.add_argument("--gpu", default=True, type=bool, help="Use GPU")
    # Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Set the learning rate")
    parser.add_argument("--hidden_units", default=512, type=int, help="Set the number of hidden layers")
    parser.add_argument("--epochs", default=8, type=int, help="Set the number of epochs")
    # Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    parser.add_argument("--save_dir", default="checkpoint.pth", type=str, help="Choose your saving path for your model")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # TODO: Build and train your network
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    training_transforms = transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                        ])

    #validation data: Resizing, then Cropping to 224 
    validation_transforms = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                        ])

    #testing data: Resizing, then Cropping to 224 
    testing_transforms =  transforms.Compose([ 
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                        ])

    # TODO: Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir,transform=testing_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    training_loader = torch.utils.data.DataLoader(training_dataset,batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(training_dataset,batch_size=32, shuffle=True)
    testing_loader =torch.utils.data.DataLoader(training_dataset,batch_size=32, shuffle=True)

    dataloaders = {
        "training"  : torch.utils.data.DataLoader(training_dataset,batch_size=64, shuffle=True),
        "validation":torch.utils.data.DataLoader(training_dataset,batch_size=64, shuffle=True),
        "testing" : torch.utils.data.DataLoader(training_dataset,batch_size=32, shuffle=True)
    }



    #Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
    if args.arch == 'vgg16': 
        model = models.vgg16(pretrained=True)
    else: 
        print("Choose vgg16 as your model architecture")
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    input = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(input, 512)),
                                ('relu', nn.ReLU()),
                                ('dropout2', nn.Dropout(p=0.5)),
                                ('hidden', nn.Linear(512, args.hidden_units)),                       
                                ('fc2', nn.Linear(args.hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1)),
                                ]))
    model.classifier = classifier
    #enable GPU
    if args.gpu:
        model.cuda()
    else:
        print("Enable GPU")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)
    epochs = args.epochs

    gpu = args.gpu
    #start only if GPU is enabled
    if (torch.cuda.is_available()):
        for e in range(epochs):
            running_loss = 0
            counter = 0
            for phase in ["training","validation"]:
                counter += 1
                if phase == "training":
                    model.train()
                else:
                    model.eval()
                # go through the training and validation data
                for data in dataloaders[phase]:
                    images, labels = data
                    images, labels = Variable(images.cuda()), Variable(labels.cuda())
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    #go forward through network
                    output = model.forward(images)
                    loss = criterion(output, labels)     
                    
                    #go backward and optimize only if in the training phase
                    if phase == "training":
                        loss.backward()
                        optimizer.step()
                        
                    #calculate statistics   
                    running_loss += loss.item()
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
                    
                if phase =="training":
                    print("Epoch: {}/{} ".format(e+1, epochs),
                        "Training Loss: {:.4f}  ".format(running_loss/counter))
                else:
                    print("Validation Loss: {:.4f}  ".format(running_loss/counter),
                        "Accuracy: {:.4f}".format(accuracy))
                
                running_loss = 0


    # TODO: Save the checkpoint 
    model.class_to_idx = training_dataset.class_to_idx
    model.cuda()
    model.epochs = epochs

    checkpoint = {
            'model': model,
            "classifier" : classifier,
            "epochs": epochs,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "class_to_idx": model.class_to_idx,
            }
    torch.save(checkpoint, args.save_dir)

if __name__ == '__main__':
    main()

