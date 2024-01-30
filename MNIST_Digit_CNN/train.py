import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from model import CNN

from torch.autograd import Variable
from torch.optim import lr_scheduler
from sklearn.calibration import calibration_curve


BATCH_SIZE = 100
N_ITER = 2500
EPOCHS = 15
LEARN_RATE = 0.003



random_seed = 1
torch.manual_seed(random_seed)

train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), download=True)
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [48000, 12000])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)


figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

model = CNN()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

# Cross Entropy Loss 
criterion = nn.CrossEntropyLoss()

# LR scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# On GPU if possible
if torch.cuda.is_available():
    print("Model will be training on GPU")
    model = model.cuda()
    criterion = criterion.cuda()
else:
    print("Model will be training on CPU")


def fit(epoch):
    
    print("Training...")
    # Set model on training mode
    model.train()
    
    # Update lr parameter
    exp_lr_scheduler.step()
    
    # Initialize train loss and train accuracy
    train_running_loss = 0.0
    train_running_correct = 0
    train_running_lr = optimizer.param_groups[0]['lr']
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.view(BATCH_SIZE,1,28,28)), Variable(target)
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1)% 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch+1, 
                 (batch_idx + 1) * len(data), 
                 len(train_loader.dataset),
                 BATCH_SIZE * (batch_idx + 1) / len(train_loader), 
                 loss.cpu().detach().numpy())
                 )
            
    train_loss = train_running_loss/len(train_loader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_loader.dataset)    
    
    return train_loss, train_accuracy, train_running_lr


def validate(data_loader):
    
    print("Validating...")
    # Set model on validating mode
    model.eval()
    val_preds = torch.LongTensor().cuda()
    val_proba = torch.LongTensor().cuda()
    
    # Initialize validation loss and validation accuracy
    val_running_loss = 0.0
    val_running_correct = 0
    
    for data, target in data_loader:
        data, target = Variable(data.view(BATCH_SIZE,1,28,28), volatile=True), Variable(target)
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        
        val_running_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        proba = torch.nn.functional.softmax(output.data)

        val_running_correct += pred.eq(target.data.view_as(pred)).cpu().sum() 
        
        # Store val_predictions with probas for confusion matrix calculations & best errors made
        val_preds = torch.cat((val_preds, pred), dim=0)
        val_proba = torch.cat((val_proba, proba))

    val_loss = val_running_loss/len(data_loader.dataset)
    val_accuracy = 100. * val_running_correct/len(data_loader.dataset) 
    
    return val_loss, val_accuracy, val_preds, val_proba


    val_loss = val_running_loss/len(data_loader.dataset)
    val_accuracy = 100. * val_running_correct/len(data_loader.dataset) 
    
    return val_loss, val_accuracy, val_preds, val_proba

best_vacc = 0.0
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
val_preds, val_proba = [], []
train_lr = []

for epoch in range(EPOCHS):
    
    print(f"Epoch {epoch+1} of {EPOCHS}\n")
    
    train_epoch_loss, train_epoch_accuracy, train_epoch_lr = fit(epoch)
    val_epoch_loss, val_epoch_accuracy, val_epoch_preds, val_epoch_proba = validate(val_loader)
    
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    train_lr.append(train_epoch_lr)
    
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    val_preds.append(val_epoch_preds)
    val_proba.append(val_epoch_proba)
    
    if val_epoch_accuracy > best_vacc:
        best_vacc = val_epoch_accuracy
        torch.save(model.state_dict(), 'weights.pth')
        print('Weights Saved')
    
    print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}\n')
    

def plot_history():

    plt.figure(figsize = (20,15))
    
    plt.subplot(221)
    
    # summarize history for accuracy
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    
    
    plt.subplot(222)
    # summarize history for loss
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    
    plt.subplot(223)
    # summarize history for lr
    plt.plot(train_lr)
    plt.title('learning rate')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.grid()
    
    plt.show()

plot_history()