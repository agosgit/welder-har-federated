# ============================================================
# ModelHAR.py
# ============================================================

import torch
import torch.nn as nn

class ModelHAR(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input shape: (batch, time, features)
        x = x.transpose(1, 2)                # (batch, features, time)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.transpose(1, 2)                # (batch, time, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]                      # ambil output terakhir LSTM
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
