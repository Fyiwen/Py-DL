# 常用的几个包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)#self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size*sequence_length,num_classes) #self.fc=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        #c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        out,_=self.rnn(x,h0)
        #out,_=self.lstm(x,(h0,c0))
        out=out.reshape(out.shape[0],-1)
        out=self.fc(out)
        #out=self.fc(out[:,-1,:])本来默认使用所有层的隐状态，但是如果只用最后一层的话可以这么改
        return out

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
input_size=28
sequence_length=28
num_layers=2
hidden_size=256
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=2

#加载数据
train_dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True) # 先把数据集搞出来，所以从这个库里把数据集下载到本地的路径下
train_loader=DataLoader(dataset=train_dataset,betch_size=batch_size,shuffle=True)
test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True) 
train_loader=DataLoader(dataset=test_dataset,betch_size=batch_size,shuffle=True)

#初始化网络
model=RNN(input_size,hidden_size,num_layers,num_classes).to(device)

#损失函数和优化器
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#train
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader): # 这里的data和target往往是这一批次中所有的样本和对应标签
        # 如果可以的话把数据放到cuda里
        data=data.to(device=device).squeeze(1)
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


#gru的话只要给rnn改个名字，但是lstm还要再改点、



