# -*- coding: utf-8 -*-
"""
Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
Author: Clara Callenberg

Make sure the sensor board is running MatClass_liveData.bin

Adjust the serial port name to your system. 
"""
import serial 
import time
import numpy as np
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms



s = serial.Serial('COM3', 460800)   # adjust to your system


n_classes = 5

def reshape_datacube(d):
    dsq = d[:,:,0:16];
    return np.reshape(dsq, (16, 16), order = 'F')

def evaluate_datacube(d, model, classes):
    #d = torch.from_numpy(d)
    dlist = []
    dlist.append(d)
    mean = 23852.534505208332
    std = 61483.19434249291
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    livedata = MaterialsDatasetLive(dlist, transform=transform)
    livedataloader = DataLoader(livedata, batch_size = 1, shuffle = True)
    with torch.no_grad():
        for data in livedataloader:
            images = data[0].to(device)
            images = images.unsqueeze(0).float()
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = classes[predicted]
            print(c)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 2 * 2, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class MaterialsDatasetLive(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
#            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)





#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load('./trainednetwork_5materials.pth', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()
classes= ('foam', 'paper', 'skin', 'towel', 'wax')

n_ROIs_x = 4;   
n_ROIs_y = 4;

bufferhistograms = 3;
histsperroi = 1;
dispCountdown = False

stillwaiting = True


datacube = np.zeros((n_ROIs_x, n_ROIs_y, 24))
datacubesMulti = np.zeros((histsperroi, n_ROIs_x, n_ROIs_y, 24))
datacubes = []
i_measurement = 0
n_measurements = 100


data = ''

while(1):
    
#    % %%%% wait until new ROI cycle begins %%%
    waitcount = 0;
    if stillwaiting:
        print('waiting for beginning of ROI cycle');

    
    while len(data) != 14 and stillwaiting:
        data = s.readline().decode('utf-8')[:-1]
        if len(data) == 14:
            if data == 'start_roicycle':
                break
        else:
            waitcount += 1
            if waitcount % 10 == 0:
                print('.')
            
    
    if stillwaiting:
        print(' starting measurement\n');
        t = time.time()
    stillwaiting = False
    
    
    data = s.readline().decode('utf-8')[:-1]
        
    
    if len(data) >= 11:
        if data[0:11] == 'ROI corners':
            histcount = 0
            roicorners = [int(s) for s in re.findall(r'\d+', data)]

    
    if len(data) > 5:
        if data[-1] == ']':
            histcount += 1;
            histo = [int(s) for s in re.findall(r'\d+', data)]
            histo = histo[1:]
            if histcount > bufferhistograms and sum(roicorners) >= 0:

                if n_ROIs_x == 13:
                    resize = 1;
                elif n_ROIs_x ==4:
                    resize = 4
                
                datacube[math.ceil(roicorners[3]/resize) , math.ceil(roicorners[0]/resize), :] = histo
                
                datacubesMulti[histcount - bufferhistograms - 1, :, :, :] = datacube;
                

                if roicorners[3] == 0 and roicorners[0] == 12 and histcount == bufferhistograms + histsperroi:
                    evaluate_datacube(reshape_datacube(datacube), model, classes)
                    for h in range(0, histsperroi):
                        datacubes.append(datacube)
                    i_measurement += 1
                    if i_measurement >= n_measurements:
                        print('Acquisition ended after %d measurements' % n_measurements)
                        break
                    datacube = np.zeros((n_ROIs_x, n_ROIs_y, 24));
                    t = time.time()
                    
                    



#%%
s.close()