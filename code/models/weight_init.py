import torch

import os
import sys

def initialize_weights(model, init = "xavier"):    
    init_func = None
    if init == "xavier":
        init_func = torch.nn.init.xavier_normal_
    elif init == "kaiming":
        init_func = torch.nn.init.kaiming_normal_
    elif init == "gaussian" or init == "normal":
        init_func = torch.nn.init.normal_

    if init_func is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Linear):
                init_func(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    else:
        print("Error when initializing model's weights, {} either doesn't exist or is not a valid initialization function.".format(init), \
            file=sys.stderr)