import custom_tools.FileIO as FileIO
import numpy as np
import os
import glob
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable

def split_dataset():
    path = '../../datasets/dogs-vs-cats/'

    # 특정 폴더 안에 들어있는 모든 파일들을 가져올때 사용 -> 확장자를 통해 가져올 데이터에 대한 확장자를 정의할 수 있음
    files = glob.glob(os.path.join(path, '*/*.jpg'))
    no_of_images = len(files)
    print('no_of_images : {}'.format(no_of_images))

    # 데이터 집합을 만드는데 사용할 셔플 색인 생성
    shuffle = np.random.permutation(no_of_images)

    # 검증 이미지를 저장할 검증용 디렉토리 생성
    FileIO.create_directory(os.path.join(path, 'valid'))

    for t in ['train', 'valid']:
        for folder in ['dog/', 'cat/']:
            FileIO.create_directory(os.path.join(path, t, folder))

    for i in shuffle[:2000]:
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'valid', folder, image))

    for i in shuffle[2000:]:
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'train', folder, image))


def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show() # 파이참에서 실행할때는 .show()까지 필요함

# 1. 학습데이터와 검증데이터 분리하기
# split_dataset()

# 2. 데이터 불러오기

# transform : 데이터 변환
simple_transform = transforms.Compose([transforms.Resize((224,224)), # 이미지 사이즈 조절
                                       transforms.ToTensor(), # 파이토치 텐서로 변화
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) # 정규화

train = ImageFolder('../../datasets/dogs-vs-cats/train/', simple_transform)
valid = ImageFolder('../../datasets/dogs-vs-cats/valid/', simple_transform)

imshow(train[50][0])

# 3. 배치 처리하기
train_data_gen = DataLoader(train, batch_size=64, num_workers=8)
valid_data_gen = DataLoader(valid, batch_size=64, num_workers=8)
dataloaders = {'train':train_data_gen,'valid':valid_data_gen}
dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}

# 4. 네트워크 아키텍쳐 구축
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2) # 프리트레인된 레즈넷 모델의 마지막 레이어를 가져와 출력 피쳐를 2로 변경

if torch.cuda.is_available():
    is_cuda = True

if is_cuda:
    model_ft = model_ft.cuda()

# 5. 오차함수와 옵티마이저 설정
learning_rate = 0.001
_criterion = nn.CrossEntropyLoss() # 오차함수
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) # 옵티마이저
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) # 학습률을 동적으로 변경

# 6. 모델 학습
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        # 까 에폭은 학습과 검증 단계로 구성
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True) # 학습 모드로 모델 선정
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반복
            for data in dataloaders[phase]:
                inputs, labels = data

                if is_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    inputs, labels = Variable(inputs), Variable(labels)

                # 파라미터 기울기 초기화
                optimizer.zero_grad()

                # 포워드
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # 학습 단계에서만 수행 (backward + optimize)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델 복사(Deep Copy)
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 최적의 모델 가중치 로딩
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, _criterion, optimizer_ft, exp_lr_scheduler, num_epochs=24)