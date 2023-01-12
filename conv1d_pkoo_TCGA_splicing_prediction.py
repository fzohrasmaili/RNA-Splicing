##########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from random import randint
import numpy as np
import time
import os
import sys
import numpy
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
##########################################################################
#Hyperparamters
data_set = sys.argv[1] #e.g. BLCA
num_epochs = 200
num_classes = 2
seq_length = 400 # or 700
learning_rate = 0.0002
##########################################################################
# Define functions
# a. Define suequence processing function
def decimal_to_binary_tensor(value_list, width):
    binary_list =[]
    for value in value_list:
        binary =np.zeros(4)
        if value == 1:
            binary = np.array([1, 0, 0, 0]) #Nucleotide A
        if value == 2:
            binary = np.array([0, 1, 0, 0]) #Nucleotide T
        if value == 3:
            binary = np.array([0, 0, 1, 0]) # Nucleotide C
        if value == 4:
            binary = np.array([0, 0, 0, 1]) # Nucleotide G
        if value == 0:
            binary = np.array([0, 0, 0, 0])
        binary_list.append(binary)
    return torch.tensor(np.asarray(binary_list), dtype=torch.uint8)

# b. Define file to list function
def file_to_list (filename):
    file_list=[]
    file=open(filename, 'r')
    cnt=1
    for line in file:
        line =line.strip().replace('n','0')
        lst=list(map(int,line))
        new_tens = decimal_to_binary_tensor(lst, width=4)
        file_list.append(new_tens)
        cnt=cnt+1
    stacked_tensor = torch.stack(file_list)
    return stacked_tensor
##########################################################################
#a. Load and Process  data
data_X = file_to_list(f"data/{data_set}/events_X.txt").float()
data_size = int(data_X.size()[0])
data_X_new= torch.transpose(data_X,1,2)
data_X_CNN = data_X_new.view(data_size,4,seq_length)
data_Y = torch.from_numpy(numpy.loadtxt(f"data/{data_set}/events_Y.txt")).long()
print (data_X.size())
################################################################################
# Define Neural Network
class ConvNet (nn.Module):
    def __init__ (self):
        super(ConvNet, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv1d (in_channels =4, out_channels=30, kernel_size=19),
            nn.ReLU(),
            nn.MaxPool1d(25),
            nn.Dropout(0.1))
        self.secondConv = nn.Sequential(
            nn.Conv1d (in_channels =30, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.1))
        self.flat=nn.Flatten()
        self.firstDense = nn.Sequential (
            nn.Linear(256,512) ,
            nn.ReLU(),
            nn.Dropout(0.5))
        self.secondDense = nn.Sequential (
            nn.Linear(512,2))
        self.sigmoid= nn.Sigmoid ()
    def forward (self, data):
        residual = data
        out = self.firstConv(data)
        out = self.secondConv(out)
        out = self.flat(out)
        out = self.firstDense(out)
        out = self.secondDense(out)
        out = self.sigmoid(out)
        return (out)
################################################################################
#Create network
convnet = ConvNet()
model =convnet
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(convnet.parameters(),lr=0.0003,  weight_decay=1e-6)
################################################################################
# Define testing function
def eval_on_test_set(model, test_data, test_label):
    running_error=0
    num_batches=0
    running_acc = 0
    data_size=(test_data.size()[0])
    testing_size = data_size
    minibatch_data = test_data
    minibatch_label = test_label
    inputs = minibatch_data.view(data_size,4,seq_length)
    scores = model(inputs)
    score_array = scores.data.cpu().numpy()
    error = utils.get_error (scores, minibatch_label)
    running_error += error.item()
    accuracy = utils.get_accuracy(scores.detach(),minibatch_label)
    running_acc += accuracy.item()
    pred_scores = scores[:,1]
    auc_roc = roc_auc_score(minibatch_label.detach().numpy(), pred_scores.detach().numpy())
    num_batches +=1
    total_error = running_error/num_batches
    total_accuracy = 100*(running_acc/num_batches)
    print ( 'Testing','\t accuracy=', total_accuracy , '\t error=', total_error,'\t auc=', auc_roc )
    return (running_error, running_acc)
################################################################################
# Training the neural network
# a. Split dataset into validation and training
Train_X, Test_X, Train_Y, Test_Y = train_test_split(data_X_CNN, data_Y, test_size=0.2, random_state=42)
train_data, val_data, train_label, val_label = train_test_split( Train_X, Train_Y, test_size=0.2, random_state=42)
#b. Train model
start = time.time()
x =[]
err =[]
acc =[]
for epoch in range (1000):
    running_loss=0
    running_error=0
    running_acc=0
    num_batches=0
    correct=0
    data_size=(train_data.size()[0])
    training_size = data_size
    batch_size =20
    for count in range (0, training_size, batch_size):
        optimizer.zero_grad()
        minibatch_data = train_data[count:count+batch_size]
        minibatch_label = train_label [count:count+batch_size]
        batch_size = (minibatch_data.size()[0])
        inputs = minibatch_data.view(batch_size,4,seq_length)
        scores = model(inputs)
        predicted = torch.argmax(scores.data, dim=1)
        loss = criterion(scores, minibatch_label)
        #Add L2 regularization
        l2_lambda = 0.000001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        loss.backward()
        optimizer.step()
        running_loss+=loss.detach().item()
        error=utils.get_error(scores.detach(),minibatch_label)
        running_error +=error.item()
        accuracy = utils.get_accuracy(scores.detach(),minibatch_label)
        running_acc += accuracy.item()
        num_batches +=1
        total_accuracy = 100*(running_acc/num_batches)
        total_loss = 100*(running_loss/num_batches)
        total_error = 100*(running_error/num_batches)
        elapsed_time = time.time()-start
        x.append(epoch)
        acc.append(total_accuracy)
        err.append(total_error)
    if epoch %50 ==0 :
        input_data = train_data.view(data_size,4,seq_length)
        all_scores = model(input_data)
        pred_scores = all_scores[:,1]
        print (pred_scores.size())
        auc_roc = roc_auc_score(train_label.detach().numpy(), pred_scores.detach().numpy())
        print ('Epoch', epoch)
        print ('Training', '\t accuracy=', total_accuracy ,
                 '\t loss=', total_loss , '\t error=', total_error , '\t auc=', auc_roc)
        print ('Validation')
        eval_on_test_set (model, val_data, val_label)
        print ('Validation on test')
        eval_on_test_set (model, Test_X, Test_Y)
        print ("*****************************************************")
print ("Final model on training")
test_error, test_correct = eval_on_test_set (model, Train_X, Train_Y)
print ("Final model on testing")
test_error, test_correct = eval_on_test_set (model, Test_X, Test_Y)

################################################################################
# Saving the trained model
model_PATH = f"data/{data_set}/Conv1D_pkoo_peak_prediction_model.pt"
torch.save(convnet.state_dict(),model_PATH)
