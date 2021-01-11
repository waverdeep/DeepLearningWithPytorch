from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Chapter05.cat_vs_dog.networks as networks
import torch

# cuda 사용 가능 확인
is_cuda=False
if torch.cuda.is_available():
    is_cuda = True

# transform : 데이터 변환
simple_transform = transforms.Compose([transforms.Resize((224,224)), # 이미지 사이즈 조절
                                       transforms.ToTensor(), # 파이토치 텐서로 변화
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) # 정규화

train = ImageFolder('../../datasets/dogs-vs-cats/train/', simple_transform)
valid = ImageFolder('../../datasets/dogs-vs-cats/valid/', simple_transform)

# 3. 배치 처리하기
train_data_gen = DataLoader(train, batch_size=16, num_workers=8, shuffle=True)
valid_data_gen = DataLoader(valid, batch_size=16, num_workers=8, shuffle=True)

# 모델 학습 함수 만들기
def fit(epoch, model, data_loader, optimizer, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    elif phase == 'validation':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)

        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # 손실 함수

        running_loss += F.nll_loss(output, target, size_average=False).data
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)

    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy


_model = networks.Net()
if is_cuda:
    _model.cuda()

_optimizer = optim.SGD(_model.parameters(), lr=0.01, momentum=0.1)
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(1, 100):
    epoch_loss, epoch_accuracy = fit(epoch, _model, data_loader=train_data_gen, optimizer=_optimizer, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, _model, valid_data_gen, _optimizer, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()
plt.show()

plt.plot(range(1,len(train_accuracy)+1),train_accuracy,'bo',label = 'train accuracy')
plt.plot(range(1,len(val_accuracy)+1),val_accuracy,'r',label = 'val accuracy')
plt.legend()
plt.show()