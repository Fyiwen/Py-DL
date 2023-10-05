import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from Dataset import CatsAndDogsDataset

device=torch.device('cude'if torch.cuda.is_available()else 'cpu')
in_channals=3
num_classes=10
learning_rate=1e-3
batch_size=32
num_epochs=1

dataset=CatsAndDogsDataset(csv_file='cats_dogs.csv',root_dir='cats_dogs_resized',transform=transforms.ToTensor())
train_set,test_set=torch.utils.data.random_split(dataset,[20000,5000]) #将数据集一共25000样本随机分割成20000和5000的训练集测试集
train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

model=torchvision.models.googlenet(pretrained=True)
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range (num_epochs):
    losses=[]
    for batch_idx,(data,targets) in enumerate(train_loader):
        data=data.to(device=device)
        targets=targets.to(device=device)
        scores=model(data)
        loss=criterion(scores,targets)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Cost at epoch{epoch} is {sum(losses)/len(losses)}')

def check_accuracy(loader,model):
    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
check_accuracy(train_loader,model)
