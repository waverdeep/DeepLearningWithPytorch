from torch.nn import ReLU
import torch

sample_data = torch.Tensor([[1,2,-1,-2]])
myReLU = ReLU()
print(myReLU(sample_data))