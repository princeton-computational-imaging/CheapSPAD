# Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
# Author: Clara Callenberg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import ListedColormap


n_classes = 5
datasets = ['foam_pink', 'paper_white', 'skin_1', 'towel_white', 'wax_white'] 		# "original colors"
# datasets = ['foam_green', 'paper_colored', 'skin_2', 'towel_blue', 'wax_red']		# color variants
# datasets = ['foam_pink_ambient', 'paper_white_ambient', 'skin_1_ambient', 'towel_white_ambient', 'wax_white_ambient']		# "original colors" with ambient light


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 2 * 2, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MaterialsDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


def load_data():
    labels = []
    data = []
    datatest = []
    labelstest= []
    count = 0
    for dataset in datasets:
        d = sio.loadmat("./data/%s.mat" % (dataset))['ds']
        tc = 0
        for i in range(0, d.shape[2]):
            if tc < 35:             # manually select last 5 positions as test data for proper training / test separation
                data.append(d[:,:,i].astype(float))
                labels.append(count)
                tc += 1
            elif tc >= 35 and tc < 40:
                datatest.append(d[:,:,i].astype(float))
                labelstest.append(count)
                if tc == 39:
                    tc = 0;
                else:
                    tc += 1
        count += 1
            
    return data, labels, datatest, labelstest


net = Net()

data, labels, datatest, labelstest = load_data()


mean = np.mean(data)
std = np.std(data)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

dataset = MaterialsDataset(data, labels, transform=transform)
datasettest = MaterialsDataset(datatest, labelstest, transform=transform)

trainset = dataset;
testset = datasettest;

trainloader = DataLoader(trainset, batch_size = 5, shuffle = True, num_workers = 0)

testloader = DataLoader(testset, batch_size = 5, shuffle = True, num_workers = 0)

classes = ('foam', 'paper', 'skin', 'towel', ' wax')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_loaders = {"train": trainloader, "val": testloader}


#%%

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)

net.to(device)

criterion = nn.CrossEntropyLoss()
loss_values = []
loss_values_valid =[]
net.train()

for epoch in range(300): # training loop

    
    for phase in ['train', 'val']:
        if phase == 'train':
            optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
            net.train(True)  # Set model to training mode
        else:
            net.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        for i, data in enumerate(data_loaders[phase], 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
            # loss.backward()
            # optimizer.step()
            running_loss += loss.item()
            
        if phase == 'train':
            loss_values.append(running_loss / len(trainset))
        else:
            loss_values_valid.append(running_loss / len(testset))

        ax.cla()
        ax.plot(loss_values, label='train loss')
        ax.plot(loss_values_valid, label='test loss')
        ax.legend()
        plt.pause(0.05)
        fig.canvas.draw()
        
plt.show()
# 

print('Finished Training')

#%% print overall accuracy

correct = 0
total = 0
with torch.no_grad():
    for d in testloader:
        images, labels = d[0].to(device), d[1].to(device)
        images = images.float()
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

#%% print class accuracy

class_correct = list(0. for i in range(n_classes))
class_total = list(0. for i in range(n_classes))
with torch.no_grad():
    for d in testloader:
        images, labels = d[0].to(device), d[1].to(device)
        images = images.float()
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(5):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(n_classes):
    print('Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    
#%% save network

torch.save(net.state_dict(), './trainednetwork.pth')


#%% plot confusion matrix

confusion_matrix = torch.zeros(n_classes, n_classes)
with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs.float())
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

pcolmap = sio.loadmat("../colormap_whitetoblack_yellow_green.mat")['map4']
pcm = ListedColormap(np.flipud(pcolmap))

plt.rcParams.update({'font.size': 20})

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
im = ax.imshow(confusion_matrix, cmap = pcm)
ax.set_xticks(np.arange(n_classes))
ax.set_yticks(np.arange(n_classes))
ax.set_xticklabels(classes, rotation='vertical', fontsize=14)
ax.set_yticklabels(classes, fontsize=14)
ax.set_xlabel("Predicted Class", labelpad=15)
ax.set_ylabel("True Class")

for i in range(n_classes):
    for j in range(n_classes):
        if confusion_matrix[i, j] > 100: 
            c = "w" 
        else: 
            c = "black"
        text = ax.text(j, i, "%d" % (confusion_matrix[i, j]), ha="center", va="center", color=c , fontsize=13)
        
plt.subplots_adjust(bottom=0.25, left=0.25)