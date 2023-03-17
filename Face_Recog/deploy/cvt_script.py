
#   TorchScript tutorial: https://pytorch.org/docs/master/jit.html

import torch
from models import ConvAngular


if __name__ == "__main__":
    #   Load vanila model
    model = ConvAngular(loss_type = 'arcface')
    model.load_state_dict(torch.load('models/model.pth'))
    #   Get part of model as for prediction
    pred_model = model.conv_block
    pred_model.eval()
    x = torch.randn(1, 3, 224, 224)
    #   Convert vanila model to torch script
    script_mode = torch.jit.trace(pred_model, x)
    torch.jit.save(script_mode, 'models/script_model.pt')
    
    