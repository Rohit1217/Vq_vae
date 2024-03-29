# -*- coding: utf-8 -*-
"""datasets_gen.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Wjufyt455sFQcEsx_tI_2chx_eb6K5vy
"""

import torch
from torchvision import transforms,datasets
from torch.utils.data import TensorDataset

def get_mnist_(quantize=256,normalize=False,device='cpu',shift=False,flatten=False):

  transform=transforms.Compose([transforms.ToTensor()])
  trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
  b,t,c=trainset.data.shape
  quant=(256.0/quantize)
  data=(trainset.data)//quant


  if normalize:
    mean = quantize//2
    var =  quantize-quantize//2
    data= (data-mean)/var

  if flatten:
    data=data.view(b,t*c)

  if shift:
    new_column=torch.full((b,1),((-1)//quantize))
    data = torch.cat((new_column, data[:, :-1]), dim=1)

  data=data.to(device)

  return data