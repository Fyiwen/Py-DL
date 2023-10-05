import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler,DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

#解决不平衡数据集的方法
#1.oversampling过采样法。还有一个欠采样就是丢弃多的那一部分，感觉不太常用。但是不管哪种都有过拟合的风险，最好搭配正则化使用。过采样有一个SMOTE方法，核心在于用相似样本合成新样本。当然这里用的这个不是
#2.class weighting类别权重

#2.指定类别的权重。比如这里把第一个类别的权重设为1（因为他的样本多），第二个类别的权重设为50（因为他样本少）。这样模型就不会一味偏向样本多的类别
loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1,50]))
#

#1.过采样。这里的操作是虽然样本少，但是经常把他加载出来用，一人用出千军万马的感觉
def get_loader(root_dir,batch_size):
    my_transforms=transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
    )
    dataset=datasets.ImageFolder(root=root_dir,transforms=my_transforms) # 从指定路径加载数据集，并且对数据集内容做出一些处理
    #
    class_weights=[1,50] # 这里是直接设定类别权重
    #
    #这里的类别权重是通过计算文件数量得到
    class_weights=[]
    for root,subdir,files in os.walk(root_dir):# 因为每个子路径下对应的是一个类别的数据，所以从files就可以得到某一个类别的样本数量
        if(len(files)>0):
            class_weights.append(1/len(files))
    #
    sample_weights=[0]*len(dataset)

    for idx,(data,label) in enumerate(dataset):# 把每个样本对应的类别权重，当作对应的每个样本权重
        class_weight=class_weights[label]
        sample_weights[idx]=class_weight
    
    sampler=WeightedRandomSampler(sample_weights,num_samples=len(sample_weights),replacement=True) # 按照样本权重，对样本进行加权采样，并且有放回
    loader=DataLoader(dataset,batch_size=batch_size,sampler=sampler) # 将采样结果作为加载出去的数据
    return loader

def main():
    loader=get_loader(root_dir="dataset",batch_size=8)
    for data,labels in loader:
        print(labels)
    num_retrievers=0
    num_elkhounds=0
    for epoch in range(10):
        for data,labels in loader: #这个输出一下就可以看到，经过上面的操作，每次的对两个类别的样本提取都很均匀
            num_retrievers+=torch.sum(labels==0)
            num_elkhounds+=torch.sum(labels==1)

    print(num_elkhounds)
    print(num_retrievers)
if __name__=="__main__":
    main()