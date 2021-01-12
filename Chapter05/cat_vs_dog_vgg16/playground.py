from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

# cuda 사용 가능 확인
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

# transform : 데이터 변환
simple_transform = transforms.Compose([transforms.Resize((224,224)), # 이미지 사이즈 조절
                                       transforms.ToTensor(), # 파이토치 텐서로 변화
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) # 정규화

# train = ImageFolder('../../datasets/dogs-vs-cats/train/', simple_transform)
valid = ImageFolder('../../datasets/dogs-vs-cats/valid/', simple_transform)

# 데이터 어구먼테이션을 위한 transformation 변경
train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

train = ImageFolder('../../datasets/dogs-vs-cats/train/', train_transform)


# 3. 배치 처리하기
train_data_loader = DataLoader(train, batch_size=16, num_workers=8, shuffle=True)
valid_data_loader = DataLoader(valid, batch_size=16, num_workers=8, shuffle=True)

# 전이학습을 위한 학습된 vgg16 모델 로드
vgg = models.vgg16(pretrained=True)
print(vgg)
vgg = vgg.cuda()

# 가중치를 업그레이드 하지 못하게 함
for param in vgg.features.parameters():
    param.requires_grad = False

# 마지막 레이어의 출력을 2로 변경
vgg.classifier[6].out_features = 2

optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.2)

#---------------#
# VGG 의 Classifier 모듈의 드롭아웃 값을 0.5에서 0.2로 변경하는 코드
#---------------#
for layer in vgg.classifier.children():
    if(type(layer) == nn.Dropout):
        layer.p = 0.2
#---------------#


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        running_loss += F.cross_entropy(output, target, size_average=False).data
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss.item() / len(data_loader.dataset)
    accuracy = 100. * running_correct.item() / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,10):
    epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)