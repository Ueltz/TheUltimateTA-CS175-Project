"""
This file contains the child of NN, MLP. MLP specifies what the NN is exactly by creating the layers, specifying the loss function, and setting the optimizer.

Contains heavily modified code:
https://www.youtube.com/watch?v=tJ3-KYMMOOs - (https://github.com/LukeDitria/pytorch_tutorials/blob/main/section03_pytorch_mlp/solutions/Pytorch1_MLP_Function_Approximation_Solution.ipynb)

Written By: Alexander Lenz
"""

import torch
from NN import NN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(NN, torch.nn.Module):
    def __init__(self, in_features, out_features, cells):
        super(MLP, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(in_features, cells[0]))
        layers.append(torch.nn.ReLU())
        for i in range(len(cells) - 1):
            layers.append(torch.nn.Linear(cells[i], cells[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(cells[-1], out_features))
        layers.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*layers)

        self.loss_function = torch.nn.MSELoss()
        #loss_function = torch.nn.L1Loss()
        #self.loss_function = torch.nn.SmoothL1Loss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)


    def get_name(self):
        return "MLP"
        