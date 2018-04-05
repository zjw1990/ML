#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:35:33 2018

@author: zhao
"""
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
#para 
SEQ_LENGTH = 28
INPUT_SIZE = 28
NUM_LAYDERS = 2
HIDDEN_SIZE = 128
NUM_CLASS =10
BATCH_SIZE = 100
NUM_EPOCHS = 2
LEARNING_RATE = 0.01


#data prepare

train_set = datasets.MNIST(root = './data/', train = True, transform = transforms.ToTensor(), download = True)
test_set = datasets.MNIST(root = './data/', train = False, transform = transforms.ToTensor(), download = True)
 
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = BATCH_SIZE, shuffle=True)

class BiLstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_class):
        super(BiLstm,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first = True, bidirectional = True)
        self.fc = nn.Linear(hidden_size*2, num_class)
    
    def forward(self,x):
        h0 = Variable(torch.zeros(self.num_layers*2 ,x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers*2 ,x.size(0), self.hidden_size)).cuda()
        
        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:, -1, :])
        
        return out


biLstm = BiLstm(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYDERS,NUM_CLASS)
biLstm.cuda()

#loss

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(biLstm.parameters(),lr=LEARNING_RATE)
#train
for epoch in range(NUM_EPOCHS):
    for i,(images,labels) in enumerate(train_loader):
        images = Variable(images.view(-1,SEQ_LENGTH,INPUT_SIZE)).cuda()
        labels = Variable(labels).cuda()
        
        
        optimizer.zero_grad()
        outputs = biLstm.forward(images)
        
        losses = loss(outputs, labels)
        
        losses.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, NUM_EPOCHS, i+1, len(train_set)//BATCH_SIZE, losses.data[0]))
        
            



# test

total = 0
correct = 0
for images,labels in test_loader:
    images = Variable(images.view(-1,SEQ_LENGTH,INPUT_SIZE)).cuda()
    outputs = biLstm.forward(images)
    
    _,predictions = torch.max(outputs.data,1)
    
    total += labels.size(0)
    correct += (predictions.cpu() == labels).sum()




print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total)) 












    
    