
import torch 
import sys
import torch.nn as nn

class LossFuncs(nn.Module):
    def __init__(self, in_features, no_classes, loss_type, s = None, m = None):
        super(LossFuncs, self).__init__()
        self.in_features = in_features
        self.no_classes = no_classes
        self.loss_func = loss_type
        self.eps = 1e-7 
        
        #   Define "s" and "m" for each model
        if loss_type == 'arcface':
            self.s = 66.0 if s is None else s
            self.m = 0.55 if m is None else m        
        if loss_type == 'cosface':
            self.s = 30.0 if s is None else s
            self.m = 0.5 if m is None else m
            
        self.fcHead = nn.Linear(in_features, no_classes, bias = False)
        
    def forward(self, x, labels):
        """
        Args:
             x: matrix of "embedding vectors" with shape: (batch_size, features_dim)
             labels: ground truth values with shape: (batch_size,)
        """
        
        #   Normalize input weights and features
        for w in self.fcHead.parameters():
            w = torch.nn.functional.normalize(w, p = 2, dim = 1)
        x = torch.nn.functional.normalize(x, p = 2, dim = 1)    #   Shape: (N, 3)
        #   Last layer with input_shape: (N, embeds_dim) and output_shape: (N, 10)
        x = self.fcHead(x)
        #   Define loss function
        if self.loss_func == 'cosface':
            num = self.s * (torch.diagonal(x[:, labels]) - self.m)
        if self.loss_func == 'arcface':
            num = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(x[:, labels]), -1.0 + self.eps, 1 - self.eps)) + self.m)   #  Clamp for numerical stability
        
        excl = torch.cat([torch.cat((x[i, :y], x[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim = 0)
        den = torch.exp(num) + torch.sum(torch.exp(self.s * excl), dim = 1)
        return -torch.mean(num - torch.log(den)), x   