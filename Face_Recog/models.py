
import torch.nn as nn
from torchvision import models
from loss_funcs import LossFuncs


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

class ConvNet(nn.Module):
    def __init__(self, embeds_dim):
        super(ConvNet, self).__init__()
        self.base_model = models.vgg16(pretrained = True)
        for param in self.base_model.parameters():
            param.requires_grad = False    
        num_ftrs = self.base_model.classifier[0].in_features
        self.base_model.classifier = Identity()
        self.fc1 = nn.Linear(num_ftrs, 4096, bias = False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(4096, embeds_dim)
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(self.relu(self.fc1(x)))
        embeds = self.relu(self.fc2(x))
        return embeds
    
    
class ConvAngular(nn.Module):
    def __init__(self, loss_type, embeds_dim = 256, no_classes = 10):
        super(ConvAngular, self).__init__()
        self.conv_block = ConvNet(embeds_dim)
        self.add_loss = LossFuncs(embeds_dim, no_classes, loss_type)
        
    def forward(self, x, labels = None, return_embedding = False):
        embeds = self.conv_block(x)
        if return_embedding:
            return embeds
        L = self.add_loss(embeds, labels)
        return L

