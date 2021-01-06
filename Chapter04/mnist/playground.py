import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# cuda 사용 가능 확인
is_cuda=False
if torch.cuda.is_available():
    is_cuda = True

# 데이터 불러오기
transformation = transforms.Compose([transforms.ToTensor(), # 텐서 형태로 변경
                                     transforms.Normalize((0.1307,), (0.3081,))]) # normalization 진행

train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

