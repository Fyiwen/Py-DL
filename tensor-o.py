import torch

#--------tensor about math and operation-------#
x=torch.tensor([1,2,3])
y=torch.tensor([9,8,7])

z1=torch.empty(3) # 大小为3，元素未知
torch.add(x,y,out=z1) # z1=[10,10,10]，x和y的对应元素相加结果存z1
z2=torch.add(x,y) # 等价。只是z1必须预先存在，z2可以是新的
z=x+y # 等价

z=torch.true_divide(x,y) #x元素除对应y，（整数和浮点都能处理）输出为浮点型，逐元素。

t=torch.zeros(3) # 形状3，全0
t.add_(x) # 与下面等价。_为原地操作符会改变调用它的张量而不产生新变量
t+=x

z=x.pow(2) # 与下面等价
z=x**2

z=x>0
z=x<0

x1=torch.rand((2,5))
x2=torch.rand((5,3))
x3=torch.mm(x1,x2) # 做二维矩阵乘法，高维要用matmal
x3=x1.mm(x2) # 等价
x3=x1@x2 # 等价

matrix_exp=torch.rand(5,5)
print(matrix_exp.matrix_power(3)) # 计算矩阵的三次方幂

z=torch.dot(x,y) # 点积，对应元素相乘后相加。必须是一维数据且长度相同

batch=32
n=10
m=20
p=30

tensor1=torch.rand((batch,n,m))
tensor2=torch.rand((batch,m,p))
out_bmm=torch.bmm(tensor1,tensor2) # (batch,n,p)批量矩阵乘法，一个批次中对应矩阵相乘

# 广播机制
#果两个张量的维度个数不同，将维度较少的张量用适当数量的前缀1来扩展，使得维度个数相同。
#如果某个张量在某一个维度上的大小为1，而另一个张量在该维度上的大小大于1，那么在该维度上将该张量扩展为与另一个张量相同的大小。
#如果两个张量在某一个维度上的大小既不相等，也不为1，则会引发错误，无法进行自动扩展。
x1=torch.rand((5,5))
x2=torch.rand((1,5))

z=x1-x2 #（5，5）
z=x1**x2 # （5，5）x1中元素取到x2中每个对应元素的幂

sun_x=torch.sum(x,dim=0) #(m,n)->(n),在0维，按行求和，每行相加，最后行没只剩列
values,indices=torch.max(x,dim=0) # 每列上最大的元素，和他按照列从上往下看的索引或者说是行索引
values,indices=torch.min(x,dim=0)
abs_x=torch.abs(x) # 对每个元素取绝对值
z=torch.argmax(x,dim=0) # 每列最大值的行索引
z=torch.argmin(x,dim=0)
mean_x=torch.mean(x.float(),dim=0) # 每一列中元素相加求个平均值
z=torch.eq(x,y) # 逐元素比较是否相等，相等就对应位置给true。保证输入形状一定一样
sort_y,indices=torch.sort(y,dim=0,descending=False) # 一个列上元素互相比较排序，索引是他们现在好了但是没排序前的行索引

z=torch.clamp(x,min=0,max=10) # 裁剪元素，使得元素范围在[min,max]中。超过的改成max，小的改成min。也可以只约束一边

x=torch.tensor([1,0,1,1,1],dtype=torch.bool) # 相当于[T，F，T，T，T]
z=torch.any(x) # 判断张量中的元素是否存在至少一个为True的值，存在就返回T


batch_size=10
feature=25
x=torch.rand((batch_size,feature))

print(x[0].shape) # （feature）
print(x[:,0].shape) # （batch_size）
print(x[2,0:10]) # (10)

x[0,0]=100

x=torch.arange(10) # [0,1,2,3,4,5,6,7,8,9]
indices=[2,5,8]
print(x[indices]) # x[2],x[5],x[8]

x=torch.rand((3,5))
rows=torch.tensor([1,0]) # 给出了想要取的元素的行索引
cols=torch.tensor([4,0]) # 给了想要取的元素的列索引
print(x[rows,cols].shape) # (2),获取的是x[1,4],x[0,0]两个元素

x=torch.arange(10)
print(x[(x<2)&(x>8)]) # 返回x中同时满足小于2和大于8的元素
print(x[x.remainder(2)==0]) # 使用取余函数，x中能被2整除的元素输出

print(torch.where(x>5,x,x*2)) # 输出x中所有函数，但是其中大于5的元素直接输出，不满足的乘2再输出
print(torch.tensor([0,0,1,2,2,3,4]).unique())# 输出张量中所有值，但是重复的不输出
print(x.ndimension()) # （1）张量的维度个数
print(x.numel()) #（10)张量中元素个数


x=torch.arange(9)

x_3x3=x.view(3,3) #重塑
x_3x3=x.reshape(3,3) #重塑

y=x_3x3.t()
y.contiguous().view(9) # 让y在内存中连续存储，对某些计算和操作可能更加高效

x1=torch.rand((2,5))
x2=torch.rand((2,5))
print(torch.cat((x1,x2),dim=0).shape) #（4，5）在第一个维度拼接，行上增加，所以第一个维度值会改变
print(torch.cat((x1,x2),dim=1).shape) # （2，10）# 在第二个维度拼接，列上增加

z=x1.view(-1) # 重塑x1变成一维张量
 
batch=64
x=torch.rand((batch,2,5))
z=x.view(batch,-1) # 也是重塑形状，不过只给定一个维度的形状，其他自动算，反正得是个二维的

z=x.permute(0,2,1) # 更换维度

x=torch.arange(10)
x.unsqueeze(0) # 不改变x，会返回的新张量形状为（1，10）
x.unsqueeze(1) # （10，1）

x=torch.arange(10).unsqueeze(0).unsqueeze(1) # （1，1，10）

z=x.squeeze(1) # 在维度1做压缩操作，但是这个维度的值必须为1，否则不进行压缩操作
