import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig


class Net(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Net, self).__init__()
        in_channels = cfg.model.in_channels
        num_classes = cfg.model.num_classes
        hidden_size = cfg.model.hidden_size
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5, stride=1)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
