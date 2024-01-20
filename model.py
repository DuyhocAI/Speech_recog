import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np


class AudioClassifier(nn.Module):
    def __init__(self, n_classes=10):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, stride=4, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=4, padding=0)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=4, padding=0)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=4, padding=0)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9, stride=4, padding=0)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=n_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        x = self.conv5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
    

if __name__ == '__main__':
    model = AudioClassifier(n_classes=10)
    
    print(model)