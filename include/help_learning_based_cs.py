import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torchvision.utils as vutils
import torchvision.datasets as dset
import numpy as np

def fft_circulant(x, t, dtype): 
    # input length
    n = x.shape[-1]
    
    # reshape x
    x = x.unsqueeze(-1)
    x = torch.cat((x, torch.zeros(x.shape).type(dtype)), -1)
    
    # F^-1 x
    IFx = torch.ifft(x, 1, normalized=True)
    
    # Ft
    Ft = torch.fft(t, 1, normalized=True)

    # Diag(sqrt * Ft) F^-1 x
    IFx_c = torch.view_as_complex(IFx)
    Ft = torch.view_as_complex(Ft)
    DIFx = torch.view_as_real(np.sqrt(n) * IFx_c *  Ft)
    
    Cx = torch.fft(DIFx, 1, normalized=True)
    return Cx[:,0]