import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.norm1 = nn.BatchNorm1d(in_channels)
        self.activation1 = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)

        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation2 = nn.LeakyReLU(0.3)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        self.max_pooling = nn.MaxPool1d(3)

        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=1) if in_channels != out_channels else None
    
    def forward(self, x):
        out = self.norm1(x)
        out = self.activation1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.conv_proj is not None:
            x = self.conv_proj(x)
        x = out + x
        x = self.max_pooling(x)

        out = self.avg_pooling(x)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        return x * x + out
