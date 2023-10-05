import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
# 这里的自定义数据集信息，包括一个文件，文件里是所有猫和狗的文件，
#一个csv文件也就是excel文档，第一列是图片的名称，第二列是图片所对应的标签

class CatsAndDogsDataset(Dataset): # 简单处理一下给予的数据，构造成一个比较标准的数据集
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self,index):# 按照索引顺序，得到对应图片+标签
        img_path=os.path.join(self.root_dir,self.annotations.iloc[index,0]) #给定行索引，列索引得到对应数据
        image=io.imread(img_path)
        y_label=torch.tensor(int(self.annotations.iloc[index,1]))

        if self.tranformer:
            image=self.transform(image)
        return (image,y_label)