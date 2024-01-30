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
import random
import seaborn as sns
import itertools
import os
import pandas as pd
from model import CNN

from torch.autograd import Variable
from torch.optim import lr_scheduler
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


BATCH_SIZE = 1

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CNN()
model.load_state_dict(torch.load('weights.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    def prediction(data_loader):
        
        print("Infering predictions...")
        # Set model on validating mode
        model.eval()
        test_pred = torch.LongTensor()
        
        for batch_idx, data in enumerate(data_loader):
            data = Variable(data[0].view(BATCH_SIZE,1,28,28))
            
            if torch.cuda.is_available():
                data = data.cuda()
                
            output = model(data)
            
            pred = output.cpu().data.max(1, keepdim=True)[1]
            test_pred = torch.cat((test_pred, pred), dim=0)
        
        print("Completed")   
        return test_pred, output

    # predict results
    y_test_pred, output = prediction(test_loader)
    
    # Associate max probability obs with label class
    y_test_pred = y_test_pred.cpu().numpy().ravel()
    
    test_images = torch.cat([batch[0] for batch in test_loader], dim=0)
    y_test = torch.cat([batch[1] for batch in test_loader], dim=0)
    y_test = y_test.cpu().numpy().ravel()
    
print(f"Test Set Metrics:\n\nAccuracy: {accuracy_score(y_test_pred, y_test)}")
print(f"Precision: {precision_score(y_test_pred, y_test, average = 'macro')}")
print(f"Recall: {recall_score(y_test_pred, y_test, average = 'macro')}")
print(f"Macro F1 Score: {f1_score(y_test_pred, y_test, average = 'macro')}")
print(f"Micro F1 Score: {f1_score(y_test_pred, y_test, average = 'micro')}")

cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])


plt.figure(figsize=(15,12))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# Sample 25 random indices
random_indices = np.random.choice(len(test_images), size=25, replace=False)

# Create a 5x5 grid of subplots with shared axes
fig, axes = plt.subplots(5, 5, figsize=(15, 15), sharex=True, sharey=True)

# Iterate through the selected indices and plot the images
for i, ax in enumerate(axes.flat):
    image_index = random_indices[i]
    # Denormalize the image if needed (assuming normalization to [0, 1])
    image = test_images[image_index].numpy().squeeze() * 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    ax.imshow(image, cmap='gray')  # Adjust cmap if your images are color
    ax.set_title(f"True: {y_test[image_index]}  Pred: {y_test_pred[image_index]}")
    ax.axis('off')

plt.tight_layout()
plt.show()