import torch
import torch.nn as nn
from torch.nn import functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4))
        self.MaxPool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.MaxPool1(out)
        out = self.conv2(out)
        out = self.MaxPool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.MaxPool3(out)
        out = out.view(-1, 9216)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=1)

        return out

input = torch.randn((1, 3, 227, 227))

net = AlexNet()
out = net(input)

print(out)