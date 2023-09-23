# 常用的几个包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#全连接神经网络
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,num_classes)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

# 设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
input_size=784
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=1

#加载数据
train_dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True) # 先把数据集搞出来，所以从这个库里把数据集下载到本地的路径下
train_loader=DataLoader(dataset=train_dataset,betch_size=batch_size,shuffle=True)
test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True) 
train_loader=DataLoader(dataset=test_dataset,betch_size=batch_size,shuffle=True)

#初始化网络
model=NN(input_size=input_size,num_classes=num_classes).to(device)

#损失函数和优化器
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#train
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader): # 这里的data和target往往是这一批次中所有的样本和对应标签
        # 如果可以的话把数据放到cuda里
        data=data.to(device=device)
        targets=targets.to(device=device)
        #调整形状作为网络输入。把data调整为二维的，因为全连接层一般都是接受二维的。如果你非要用三维那只能展平成二维，或者用CNN了
        data=data.reshape(data.shape[0],-1)

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
