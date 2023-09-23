# 常用的几个包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#卷积神经网络
class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10): #外层的自定义相当于固定了初始的输入通道数和最终的输出的某个大小，注意彩色图片要用3个通道
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))# 从这边开始算是里面的自定义除去开头和结尾以外，其他都可以自由一点。这一层的写法比较特殊会导致输入出大小不变
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) # 使得特征图尺寸减半
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1)) # 注意通道数的对应
        self.fc1=nn.Linear(16*7*7,num_classes) # 线性层的这个前一层定义需要计算一下。虽然说卷积网络可以容纳任意大小的输入，但是如果用了全连接其实就不是任意l，得事先知道输入大小才好设计网络
    def forward(self,x):
        x=F.relu(self.conv1(x)) 
        x=self.pool(x) 
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)
        return x

def save_checkpoint(state,filename="mu_checkpoint.pth.tar"): # 因为没有完整路径，只给出了文件名，所以这个文件最终会保存在当前目录下
    print("=>Saving checkpoint")
    torch.save(state,filename) # 保存的检查点状态以这个文件名存储

def load_checkpoint(checkpoint):
    print("=>Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict']) # 加载检查点状态
    optimizer.load_state_dict(checkpoint['optimizer'])
#model=CNN()
#x=torch.randn(64,1,28,28)一共64个批次，每个批次中每一张图片如果黑白就是1*28*28，彩色就应该是3*28*28
# 设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
in_channel=1
num_classes=10
learning_rate=1e-4
batch_size=1024
num_epochs=10
load_model=True

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

if load_model:
    load_checkpoint(torch.load("my_checkpoint.path.tar"))
    
#train
for epoch in range(num_epochs):
    losses=[]
    if epoch==2:
        checkpoint={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()} # 这里只保存了模型和优化器的状态，如果还需要保留其他的，这边可以更改。:model.state_dict()本身就是个字典，包含模型的所有参数
        save_checkpoint(checkpoint)


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
