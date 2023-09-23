# 常用的几个包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
in_channel=3
num_classes=10
learning_rate=1e-3
batch_size=1024
num_epochs=5

import sys

class Identity(nn.module): # 定义一个恒等变换模型
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x):
        return x
    
# 加载以前的模型，还可以修改它
model=torchvision.models.vgg16(pretrained=True) # 加载了一个已经预训练好了的vgg16网络

for param in model.parameters(): # 这个决定了加载的这个模型里的参数要不要参加训练发生改变
    param.requires_grad=False

model.avgpool=Identity() #把网络里的这个层换成恒等映射层
model.classifier=nn.Linear(512,10) # 这个层在更换的时候要注意，512哪来，是你先得输出看一下加载的模型每一层有什么，参数是多少，需要根据那个进行修改
model.classifier=nn.Sequential(nn.Linear(512,100),nn.DropOut(p=0.5),nn.Linear(100,10))
for i in range(1,7):
    model.classifier[i]=Identity()#这种情况也要注意，因为模型中这个名字的网络不止一个，就需要索引来确定具体是哪一层
model.to(device)
sys.exit() # 这一句其实这里用不到，但是可以学一下，反正就是整个程序就运行到这里，后面的都不执行


#加载数据
train_dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True) # 先把数据集搞出来，所以从这个库里把数据集下载到本地的路径下
train_loader=DataLoader(dataset=train_dataset,betch_size=batch_size,shuffle=True)
test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True) 
train_loader=DataLoader(dataset=test_dataset,betch_size=batch_size,shuffle=True)

#初始化网络
model=CNN().to(device)

#损失函数和优化器
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#train
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader): # 这里的data和target往往是这一批次中所有的样本和对应标签
        # 如果可以的话把数据放到cuda里
        data=data.to(device=device)
        targets=targets.to(device=device)
        
        # 前向传播
        scores=model(data)
        loss=criterion(scores,targets)

        # 反向传播更新
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

#accuracy
def check_accuracy(loader,model):
    if loader.dataset.train: # 判断是测试集还是训练集
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    
    num_correct=0
    num_samples=0
    model.eval() # 主要影响BN和dropout层。例如DO层会让所有激活单元都通过，BN停止计算和更新均值和方差，但是不影响梯度，梯度依旧计算只是不进行反传

    with torch.no_grad(): # 停止梯度的计算，不影响BN和DO
        for x,y in loader: # 一次循环得到一个批次大小的所有，和上面的那种写法的差别就是没有批次索引
            x=x.to(device=device)
            y=y.to(device=device)
            x=x.reshape(x.shape[0],-1)

            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f'got {num_correct}/{num_samples} with accuracy {float(num_correct)}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader,model)
