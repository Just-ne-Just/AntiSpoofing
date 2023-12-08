import torch
import torch.nn as nn
from hw_spoof.model.sinc_conv import SincConvFast
from hw_spoof.model.res_block import ResBlock
import numpy as np

class RawNet(nn.Module):
    def __init__(self, conv_channels, sinc_kernel_size=1024, gru_hidden=1024, n_gru=3, fc_channels=1024, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sinc = SincConvFast(conv_channels[0][0], kernel_size=sinc_kernel_size)
        self.max_pooling = nn.MaxPool1d(3)
        self.norm = nn.BatchNorm1d(conv_channels[0][0])
        self.activation = nn.LeakyReLU(0.3)

        self.first_blocks = nn.ModuleList(
            [ResBlock(conv_channels[0][0], conv_channels[0][1]) for _ in range(2)]
        )

        self.middle_block = ResBlock(conv_channels[0][1], conv_channels[1][0])

        self.second_blocks = nn.ModuleList(
            [ResBlock(conv_channels[1][0], conv_channels[1][1]) for _ in range(3)]
        )

        self.gru_norm = nn.BatchNorm1d(conv_channels[1][1])
        self.gru_activation = nn.LeakyReLU(0.3)

        self.gru = nn.GRU(input_size=conv_channels[1][1],
                          hidden_size=gru_hidden,
                          num_layers=n_gru,
                          batch_first=True)
        
        
        self.fc1 = nn.Linear(gru_hidden, fc_channels)
        self.fc2 = nn.Linear(fc_channels, 2)


    def forward(self, audio, **kwargs):
        x = audio.unsqueeze(1)
        x = self.sinc(x)
        x = self.max_pooling(torch.abs(x))
        x = self.norm(x)
        x = self.activation(x)

        for first_block in self.first_blocks:
            x = first_block(x)
        
        x = self.middle_block(x)

        for second_block in self.second_blocks:
            x = second_block(x)
        
        x = self.gru_norm(x)
        x = self.gru_activation(x)
        x = x.transpose(1, 2)

        self.gru.flatten_parameters()

        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.fc2(x)

        print("Model prediction", x)
        return {"prediction": x}
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


if __name__ == "__main__":
    m = RawNet(conv_channels=[[20, 20], [128, 128]])
    print(m)
    batch = torch.randn(size=(1, 64000))
    m(batch)