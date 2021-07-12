## 深度学习记录

### 工具使用

- 开启 tensorboard

  ```bash
  #开启
  tensorboard --logdir='tensorboard的log地址' --port=6006
  #映射到本地16006端口
  ssh -L 16006:127.0.0.1:6006 user@address #用户和服务器地址
  #本地地址
  http://127.0.0.1:16006
  ```

- tensorboardX使用

  [详解PyTorch项目使用TensorboardX进行训练可视化](https://blog.csdn.net/bigbennyguo/article/details/87956434)

  - 创建一个SummaryWriter的实例

  ```python
  from tensorboardX import SummaryWriter
  
  #提供一个路径，将使用该路径来保存日志
  # Creates writer1 object.
  # The log will be saved in 'runs/exp'
  writer1 = SummaryWriter('runs/exp')
  
  #无参数，默认使用runs/日期时间路径来保存日志
  # Creates writer2 object with auto generated file name
  # The log directory will be something like 'runs/Aug20-17-20-33'
  writer2 = SummaryWriter()
  
  #提供一个comment参数，将使用runs/日期时间-comment路径来保存日志
  # Creates writer3 object with auto generated file name, the comment will be appended to the filename.
  # The log directory will be something like 'runs/Aug20-17-20-33-resnet'
  writer3 = SummaryWriter(comment='resnet')
  ```

  - 使用各种add方法记录数据

    使用add_scalar方法来记录数字常量

    > add_scalar(tag,scalar_value,global_step=None,walltime=None)
    >
    > 参数：
    >
    > - tag(string): 数据名称，不同名称的数据使用不同曲线展示
    > - scalar_value(float): 数字常量值
    > - global_step(int,optional): 训练的step
    > - walltime(falot,optional):记录发生的时间，默认为time.time()

    需要注意，这里的 scalar_value 一定是 float 类型，如果是 PyTorch scalar tensor，则需要调用 .item() 方法获取其数值。我们一般会使用 add_scalar 方法来记录训练过程的 loss、accuracy、learning rate 等数值的变化，直观地监控训练过程。

    **Example**:

    ```
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('runs/scalar_example')
    for i in range(10):
        writer.add_scalar('quadratic', i**2, global_step=i)
        writer.add_scalar('exponential', 2**i, global_step=i)
    
    ```

- 开启visdom

  ```bash
  nohup python -m visdom.server &
  #远程服务器映射到本地端口
  ssh -L 8097:127.0.0.1:8097 dell@202.120.92.171
  ```

### 数学知识

#### Sigmoid函数

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

图像：

<img src="C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210710160010776.png" alt="image-20210710160010776" style="zoom:67%;" />

求导：
$$
\sigma^{'}(z)=\frac{1}{1+e^{-z}} - \frac{1}{(1+e^{-z})^2}=\sigma(z)(1-\sigma(z))
$$

#### Softmax函数

$$
softmax(x)_i = \frac{exp(x_i)}{\sum_jexp(x_j)}
$$

### PyTorch

- tensor to numpy

  ```python
  a = torch.ones(5)
  
  b = a.numpy()
  ```

- numpy to tensor

  ```python
  a = numpy.ones(5)
  
  b = torch.from_numpy(a)
  ```

- item()

  item()函数可以理解为得到 **数值**，就是纯粹的一个值。

  ```python
  x = torch.randn(1)
  print(x)
  print(x.item())
  
  # tensor([-0.2368])
  # -0.23680149018764496
  ```

#### 数据类型转换

Pytorch中的Tensor常用的类型转换函数（inplace操作）：

（1）数据类型转换

　　在Tensor后加 .long(), .int(), .float(), .double()等即可，也可以用.to()函数进行转换，所有的Tensor类型可参考https://pytorch.org/docs/stable/tensors.html

（2）数据存储位置转换

　　CPU张量 ----> GPU张量，使用data.cuda()

　　GPU张量 ----> CPU张量，使用data.cpu()

（3）与numpy数据类型转换

　　Tensor---->Numpy 使用 data.numpy()，data为Tensor变量

　　Numpy ----> Tensor 使用 torch.from_numpy(data)，data为numpy变量

（4）与Python数据类型转换

　　Tensor ----> 单个Python数据，使用data.item()，data为Tensor变量且只能为包含单个数据

　　Tensor ----> Python list，使用data.tolist()，data为Tensor变量，返回shape相同的可嵌套的list

（5）剥离出一个tensor参与计算，但不参与求导

　　Tensor后加 .detach()

#### 冻结参数

如果整个过程完全不动这部分的参数，直接设置optimizer的时候只声明要被训练的参数就行，比如：

假设你的模型里有4个层（nn.Module），那么在设置优化器时，这样写：

```
opt = SGD([{'params':model[0].parameters()}],lr=0.1)
```

#### Dataset

torch.utils.data.Dataset是一个抽象类，用户想要加载自定义的数据只需要继承这个类，并且重写其中的两个方法即可：

- \__len__：实现 len(dataset) 返回整个数据集的大小
- \__getitem__：用来获取一些索引的数据，是 dataset[i] 返回数据集中第i个样本

```python
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```

#### DataLoader

```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, \
    batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, \
    drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
```

dataset：定义的dataset类返回的结果

batch_size：每个batch要加载的样本数，默认为1。

shuffle：在每个epoch中对整个数据集data进行shuffle重排，默认为False。

sample：定义从数据集中加载数据所采用的策略，如果指定的话，shuffle必须为False；batch_sample类似，表示一次返回一个batch的index。

num_workers：表示开启多少个线程数取加载你的数据，默认为0，代表只使用主线程。

collate_fn：表示合并样本列表以形成小批量的Tensor对象。默认的collate_fn函数是要求一个batch中的图片都具有相同size（因为要做stack操作），当一个batch中的图片大小都不同时，可以使用自定义的collate_fn函数，则一个batch中的图片不再被stack操作，**可以全部存储在一个list中**，当然还有对应的label。

pin_memory：表示要将load进来的数据是否要拷贝到pin_memory区中，其表示生成的Tensor数据是属于内存中的锁内存区，这样将Tensor数据转义到GPU中速度就会快一些，默认为False。

drop_last：当你的整个数据长度不能够整除你的batch_size，选择是否要丢弃最后一个不完整的batch，默认为False。

### 代码相关

```python
#选择指定显卡运行程序
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
```

```bash
#anaconda创建新环境
conda create -n xxx python=3.6
#anaconda删除环境
conda remove -n your_env_name(虚拟环境名称) --all
#删除没有用的包
conda clean -p   
#tar打包
conda clean -t      
#删除所有的安装包及cache
conda clean -y -all 

#####
## pytorch安装
#####
#测试是否安装成功
torch.cuda.is_available() 
```

### 卷积核相关

为什么卷积核的大小一般都是奇数

（1）奇数卷积核更容易做padding。我们假设卷积核大小为k*k，为了让卷积后的图像大小与原图一样大，根据公式可得到padding = (k-1)/2。这里的k只有在取奇数的时候，padding才能使整数，否则padding不好进行图片填充。

（2）更容易找到锚点。在CNN中，一般以卷积核的某个基准点进行窗口滑动，通常这个基准点是卷积核的中心点，所以如果k是偶数，就找不到中心点。

### 分类问题

| 分类问题名称 | 输出层使用激活函数 | 对应损失函数         |
| ------------ | ------------------ | -------------------- |
| 二分类       | sigmoid函数        | 二分类交叉熵损失函数 |
| 多分类       | softmax函数        | 多类别交叉熵损失函数 |
| 多标签分类   | sigmoid函数        | 二分类交叉熵损失函数 |

#### CrossEntropyLoss

CrossEntropyLoss函数包含Softmax层、log和NLLLoss层，适用于**单标签**任务，主要用在单标签多分类任务上，当然也可以用在单标签二分类上。

Pytorch中计算的交叉熵并不是采用：
$$
H(p,q) = -\sum_x(p(x)logq(x)+(1-p(x))log(1-q(x)))
$$
这种方式计算得到的，而是交叉熵的另外一种方式计算得到的：
$$
H(p,q)=-\sum p(x)logq(x)
$$
它是交叉熵的另外一种方式。

Pytorch中CrossEntropyLoss()函数的主要是将softmax-log-NLLLoss合并到一块得到的结果。

    1、Softmax后的数值都在0~1之间，所以ln之后值域是负无穷到0。

    2、然后将Softmax之后的结果取log，将乘法改成加法减少计算量，同时保障函数的单调性 。

    3、NLLLoss的结果就是把上面的输出与Label对应的那个值拿出来，去掉负号，再求均值。

```python
import torch
import torch.nn as nn
x_input=torch.randn(3,3)#随机生成输入 
print('x_input:\n',x_input) 
y_target=torch.tensor([1,2,0])#设置输出具体值 print('y_target\n',y_target)

#计算输入softmax，此时可以看到每一行加到一起结果都是1
softmax_func=nn.Softmax(dim=1)
soft_output=softmax_func(x_input)
print('soft_output:\n',soft_output)

#在softmax的基础上取log
log_output=torch.log(soft_output)
print('log_output:\n',log_output)

#对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
logsoftmax_func=nn.LogSoftmax(dim=1)
logsoftmax_output=logsoftmax_func(x_input)
print('logsoftmax_output:\n',logsoftmax_output)

#pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
nllloss_func=nn.NLLLoss()
nlloss_output=nllloss_func(logsoftmax_output,y_target)
print('nlloss_output:\n',nlloss_output)

#直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)
print('crossentropyloss_output:\n',crossentropyloss_output)
```

最后计算得到的结果：

```text
x_input:
 tensor([[ 2.8883,  0.1760,  1.0774],
        [ 1.1216, -0.0562,  0.0660],
        [-1.3939, -0.0967,  0.5853]])
y_target
 tensor([1, 2, 0])
soft_output:
 tensor([[0.8131, 0.0540, 0.1329],
        [0.6039, 0.1860, 0.2102],
        [0.0841, 0.3076, 0.6083]])
log_output:
 tensor([[-0.2069, -2.9192, -2.0178],
        [-0.5044, -1.6822, -1.5599],
        [-2.4762, -1.1790, -0.4970]])
logsoftmax_output:
 tensor([[-0.2069, -2.9192, -2.0178],
        [-0.5044, -1.6822, -1.5599],
        [-2.4762, -1.1790, -0.4970]])
nlloss_output:
 tensor(2.3185)
crossentropyloss_output:
 tensor(2.3185)
```

#### BCELoss 

BCELoss可以看作是CrossEntropyLoss的一个特例，适用于**二分类**任务，可以是单标签二分类，也可以是多标签二分类任务。

BCELoss的计算公式为：
$$
-\frac{1}{n}\sum(y_nlnx_n+(1-y_n)ln(1-x_n))
$$

#### BCEWithLogistsLoss

BCEWithLogitsLoss函数包括了Sigmoid层和BCELoss层，适用于**二分类**任务，可以是单标签二分类，也可以是多标签二分类任务。

```python
label = torch.Tensor([1, 1, 0])
pred = torch.Tensor([3, 2, 1])
pred_sig = torch.sigmoid(pred)
# BCELoss must be used together with sigmoid
loss = nn.BCELoss()
print(loss(pred_sig, label))
# BCEWithLogitsLoss
loss = nn.BCEWithLogitsLoss()
print(loss(pred, label))
```

输出结果：

```text
tensor(0.4963)  # BCELoss
tensor(0.4963)  # BCEWithLogiticsLoss
```

#### BatchNorm2d

批归一化：每一次优化时的样本数目，通常BN网络层用在卷积层后，用于重新调整数据分布

好处：(1) 减轻了对参数初始化的依赖，有利地解决了梯度爆炸或消失问题。

​            (2) 训练更快，可以使用更高的学习率。

​            (3) BN一定程度上增加了泛化能力，dropout等技术可以去掉。

![image-20210708211623023](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210708211623023.png)

$E(x)$和$Var(x)$为批量数据的均值和方差，$\epsilon$为了防止分母为0，对应参数eps

- num_features：一般输入参数为(N,C,H,W)，即为其中特征的数量

- eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5

- momentum：一个用于运行过程中均值和方差的一个估计参数，默认为：0.1

- affine：当设为true时，会给定可以学习的系数矩阵gamma和beta

### yolov3

#### yolov3.cfg文件说明

```
[net]                        ★ [xxx]开始的行表示网络的一层，其后的内容为该层的参数配置，[net]为特殊的层，配置整个网络
# Testing                    ★ #号开头的行为注释行，在解析cfg的文件时会忽略该行
# batch=1
# subdivisions=1
# Training
batch=64                     ★ 这儿batch与机器学习中的batch有少许差别，仅表示网络积累多少个样本后进行一次BP 
subdivisions=16              ★ 这个参数表示将一个batch的图片分sub次完成网络的前向传播
                             ★★ 敲黑板：在Darknet中，batch和sub是结合使用的，例如这儿的batch=64，sub=16表示训练的过
                             程中将一次性加载64张图片进内存，然后分16次完成前向传播，意思是每次4张，前向传播的循环过程中
                             累加loss求平均，待64张图片都完成前向传播后，再一次性后传更新参数
                             ★★★ 调参经验：sub一般设置16，不能太大或太小，且为8的倍数，其实也没啥硬性规定，看着舒服就好
                             batch的值可以根据显存占用情况动态调整，一次性加减sub大小即可，通常情况下batch越大越好，还需
                             注意一点，在测试的时候batch和sub都设置为1，避免发生神秘错误！
 
width=608                    ★ 网络输入的宽width
height=608                   ★ 网络输入的高height
channels=3                   ★ 网络输入的通道数channels 3为RGB彩色图片，1为灰度图，4为RGBA图，A通道表示透明度
                             ★★★ width和height一定要为32的倍数，否则不能加载网络
                             ★ 提示：width也可以设置为不等于height，通常情况下，width和height的值越大，对于小目标的识别
                             效果越好，但受到了显存的限制，读者可以自行尝试不同组合
                             
momentum=0.9                 ★ 动量 DeepLearning1中最优化方法中的动量参数，这个值影响着梯度下降到最优值的速度
decay=0.0005                 ★ 权重衰减正则项，防止过拟合,decay参数越大对过拟合的抑制能力越强
 
angle=5                      ★ 数据增强参数，通过旋转角度来生成更多训练样本，生成新图片的时候随机旋转-5~5度
saturation = 1.5             ★ 数据增强参数，通过调整饱和度来生成更多训练样本，饱和度变化范围1/1.5到1.5倍
exposure = 1.5               ★ 数据增强参数，通过调整曝光量来生成更多训练样本，曝光量变化范围1/1.5到1.5倍
hue=.1                       ★ 数据增强参数，通过调整色调来生成更多训练样本，色调变化范围-0.1~0.1 
 
learning_rate=0.001          ★ 学习率决定着权值更新的速度，设置得太大会使结果超过最优值，太小会使下降速度过慢。
                             如果仅靠人为干预调整参数，需要不断修改学习率。刚开始训练时可以将学习率设置的高一点，
                             而一定轮数之后，将其减小在训练过程中，一般根据训练轮数设置动态变化的学习率。
                             刚开始训练时：学习率以 0.01 ~ 0.001 为宜。一定轮数过后：逐渐减缓。
                             接近训练结束：学习速率的衰减应该在100倍以上。
                             学习率的调整参考https://blog.csdn.net/qq_33485434/article/details/80452941
                             ★★★ 学习率调整一定不要太死，实际训练过程中根据loss的变化和其他指标动态调整，手动ctrl+c结
                             束此次训练后，修改学习率，再加载刚才保存的模型继续训练即可完成手动调参，调整的依据是根据训练
                             日志来，如果loss波动太大，说明学习率过大，适当减小，变为1/5，1/10均可，如果loss几乎不变，
                             可能网络已经收敛或者陷入了局部极小，此时可以适当增大学习率，注意每次调整学习率后一定要训练久
                             一点，充分观察，调参是个细活，慢慢琢磨
                             ★★ 一点小说明：实际学习率与GPU的个数有关，例如你的学习率设置为0.001，如果你有4块GPU，那
                             真实学习率为0.001/4
burn_in=1000                 ★ 在迭代次数小于burn_in时，其学习率的更新有一种方式，大于burn_in时，才采用policy的更新方式
max_batches = 500200         ★ 训练次数达到max_batches后停止学习，一次为跑完一个batch
 
policy=steps                 ★ 学习率调整的策略：constant, steps, exp, poly, step, sig, RANDOM，constant等方式
                             参考https://nanfei.ink/2018/01/23/YOLOv2%E8%B0%83%E5%8F%82%E6%80%BB%E7%BB%93/#more
steps=400000,450000          
scales=.1,.1                 ★ steps和scale是设置学习率的变化，比如迭代到400000次时，学习率衰减十倍，45000次迭代时，学
                             习率又会在前一个学习率的基础上衰减十倍
 
[convolutional]              ★ 一层卷积层的配置说明
batch_normalize=1            ★ 是否进行BN处理，什么是BN此处不赘述，1为是，0为不是 
filters=32                   ★ 卷积核个数，也是输出通道数
size=3                       ★ 卷积核尺寸
stride=1                     ★ 卷积步长
pad=1                        ★ 卷积时是否进行0 padding,padding的个数与卷积核尺寸有关，为size/2向下取整，如3/2=1
                             # 如果pad为0,padding由 padding参数指定。如果pad为1，padding大小为size/2
activation=leaky             ★ 网络层激活函数
                             ★★ 卷积核尺寸3*3配合padding且步长为1时，不改变feature map的大小
                             
# Downsample
[convolutional]              ★ 下采样层的配置说明
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky             ★★ 卷积核尺寸为3*3，配合padding且步长为2时，feature map变为原来的一半大小
 
[shortcut]                   ★ shotcut层配置说明
from=-3                      ★ 与前面的多少次进行融合，-3表示前面第三层
activation=linear            ★ 层次激活函数包括，logistic, loggy, relu, elu, relie, plse, hardtan, lhtan, linear, 
                             ramp, leaky, tanh, stair
    ......
    ......
[convolutional]              ★ YOLO层前面一层卷积层配置说明
size=1
stride=1
pad=1
filters=255                  ★ 每一个[region/yolo]层前的最后一个卷积层中的 filters=(classes+5)*anchors_num,其中
                             5的意义是4个坐标加一个置信率，即论文中的tx,ty,tw,th,to
                             anchors_num 是该层mask的一个值.如果没有mask则 anchors_num=num
                             classes为类别数，COCO为80,num表示YOLO中每个cell预测的框的个数，YOLOV3中为3
                             ★★★ 自己使用时，此处的值一定要根据自己的数据集进行更改，例如你识别80个类，则：
                             filters=3*(80+5)=255,三个fileters都需要修改，切记
activation=linear
 
[yolo]                       ★ YOLO层配置说明
mask = 0,1,2                 ★  使用anchor的索引，0，1，2表示使用下面定义的anchors中的前三个anchor
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326   
                             ★ 预测框的初始宽高，第一个是w，第二个是h，总数量是num*2,YOLOv2作者说anchors是使用K-MEANS
                             获得，其实就是先统计出哪种大小的框比较多,可以增加收敛速度,如果不设置anchors,默认是0.5;
classes=80                   ★ 类别数目
num=9                        ★ 每个grid cell总共预测几个box,和anchors的数量一致。当想要使
                             用更多anchors时需要调大num，且如果调大num后训练时Obj趋近0的话
                             可以尝试调大object_scale
jitter=.3                    ★ 数据增强手段，此处jitter为随机调整宽高比的范围，通过抖动增加噪声来抑制过拟合
                             [?]利用数据抖动产生更多数据，YOLOv2中使用的是crop，filp，以及net层的angle，flip是随机的，
					         jitter就是crop的参数，tiny-yolo-voc.cfg中jitter=.3，就是在0~0.3中进行crop
ignore_thresh = .7
truth_thresh = 1             ★ 参与计算的IOU阈值大小.当预测的检测框与ground true的IOU大于ignore_thresh的时候，参与
                             loss的计算，否则，检测框的不参与损失计算。
                             ★ 理解：目的是控制参与loss计算的检测框的规模，当ignore_thresh过于大，接近于1的时候，那么参与
                             检测框回归loss的个数就会比较少，同时也容易造成过拟合；而如果ignore_thresh设置的过于小，那么
                             参与计算的会数量规模就会很大。同时也容易在进行检测框回归的时候造成欠拟合。
                             ★ 参数设置：一般选取0.5-0.7之间的一个值，之前的计算基础都是小尺度（13*13）用的是0.7，
                             （26*26）用的是0.5。这次先将0.5更改为0.7。参考：https://www.e-learn.cn/content/qita/804953
random=1                     ★ 为1打开随机多尺度训练，为0则关闭
                             ★★ 提示：当打开随机多尺度训练时，前面设置的网络输入尺寸width和height其实就不起作用了，width
                             会在320到608之间随机取值，且width=height，每10轮随机改变一次，一般建议可以根据自己需要修改
                             随机尺度训练的范围，这样可以增大batch，望读者自行尝试！
```

#### 目标检测评价指标

- IOU交并比

- TP (真阳性，被正确分类的正例) 

- FN(假阴性，本来是正例，错分为负例) 

- TN(真阴性，被正确分类的负例) 

- FP(假阳性，本来是负例，被错分为正例)

- 精确率P

  - P = TP/(TP+FP) 

- 召回率

  - R = TP/(TP+FN)

- AP 顾名思义AP就是平均精准度，简单来说就是对PR曲线上的Precision值求均值。对于pr曲线来说，我们使用积分来进行计算。

  ![[公式]](https://www.zhihu.com/equation?tex=AP%3D%5Cint_%7B0%7D%5E%7B1%7Dp%28r%29dr)

  - AP<sub>50</sub>：IoU阈值为0.5时的AP测量值
  - AP<sub>70</sub>：IoU阈值为0.75时的测量值
  - AP@50:5:95指的是IOU的值从50%取到95%，步长为5%，然后算在在这些IOU下的AP的均值
  - AP<sub>S</sub> : 像素面积小于 ![[公式]](https://www.zhihu.com/equation?tex=32%5E2) 的目标框的AP测量值
  - AP<sub>M</sub> : 像素面积在![[公式]](https://www.zhihu.com/equation?tex=32%5E2)- ![[公式]](https://www.zhihu.com/equation?tex=96%5E2)之间目标框的测量值
  - AP<sub>L</sub> : 像素面积大于 ![[公式]](https://www.zhihu.com/equation?tex=96%5E2) 的目标框的AP测量值

#### VOC标注转换

- 路径转换

  ```python
  import os
  
  data = 'main/test.txt'
  generate = 'test.txt'
  
  with open(data,'r') as f:
      with open(generate,'w+') as g:
        for line in f.readlines():
            line = 'data/voc/images/' + line.rstrip() +'.jpg'
            g.write(line + "\n")
  ```

- 文件转换

  ```python
  import xml.etree.ElementTree as ET
  import pickle
  import os
  from os import listdir, getcwd
  from os.path import join
   
  sets=['trainval','test']  # 我只用了VOC
   
  classes = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"]  # 修改为自己的label
   
  def convert(size, box):
      dw = 1./(size[0])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dw = 1./((size[0])+0.1)
      dh = 1./(size[1])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dh = 1./((size[0])+0.1)
      x = (box[0] + box[1])/2.0 - 1
      y = (box[2] + box[3])/2.0 - 1
      w = box[1] - box[0]
      h = box[3] - box[2]
      x = x*dw
      w = w*dw
      y = y*dh
      h = h*dh
      return (x,y,w,h)
   
  def convert_annotation(image_id):
      in_file = open('VOC/Annotations/%s.xml'%image_id)
      out_file = open('VOC/Labels/%s.txt'%image_id, 'w')
      tree=ET.parse(in_file)
      root = tree.getroot()
      size = root.find('size')
      w = int(size.find('width').text)
      h = int(size.find('height').text)
  
      obj = root.find("object")
      xmlbox = obj.find('bndbox')
      b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
      bb = convert((w,h), b)
      #return str(0) + " " + " ".join([str(a) for a in bb]) + '\n'
      out_file.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')
      
      '''
      for obj in root.iter('object'):
          difficult = obj.find('difficult').text
          cls = obj.find('name').text
          if cls not in classes or int(difficult)==1:
              continue
          cls_id = classes.index(cls)
          xmlbox = obj.find('bndbox')
          b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
          bb = convert((w,h), b)
          out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
      '''
  
  if __name__ == "__main__":
      '''
      wd = 'data/voc/images'
   
      for image_set in sets:
          if not os.path.exists('VOC/Labels/'):
              os.makedirs('VOC/Labels/')
          image_ids = open('VOC/ImageSets/Main/%s.txt'%image_set).read().strip().split()
          list_file = open('VOC/%s.txt'%(image_set), 'w')
          #out_file = open('VOC/Labels/%s.txt'%(image_set), 'w')
          for image_id in image_ids:
              list_file.write('%s/%s.png\n'%(wd,image_id))
              #out_file.write(convert_annotation(image_id))
              convert_annotation(image_id)
          list_file.close()
          #out_file.close()
      '''
      #print(convert((353,500),(48,195,240,371)))
   
  # 这块是路径拼接，暂时用不上，先都注释了
  # os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
  # os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
  ```

#### GIoU

**IoU的不足点：**

- 如果两个目标没有重叠，IoU将会为0，并且不会反应两个目标之间的距离，在这种无重叠目标的情况下，如果IoU用作损失函数，梯度为0，无法优化。

- IoU无法区分两个对象之间不同的对齐方式。更确切地讲，不同方向上有相同交叉级别的两个重叠对象的IoU会完全相等。

  ![image-20210625183258259](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210625183258259.png)

**GIoU:**

对于任意的两个Ａ、B框，首先找到一个能够包住它们的最小方框Ｃ。然后计算C \ (A ∪ B) 的面积与Ｃ的面积的比值，注：C - (A ∪ B) 的面积为C的面积减去A∪B的面积。再用Ａ、Ｂ的IoU值减去这个比值得到GIoU。

![image-20210625182601192](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210625182601192.png)
$$
IoU = \frac{|A \cap B|}{|A\cup B|} \\
GIoU = IoU - \frac{|C-(A\cup B)|}{|C|}
$$


#### Focal Loss

Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题。该损失函数降低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘。

Focal loss是在交叉熵损失函数基础上进行的修改，首先回顾二分类交叉熵损失：
$$
L = -ylogy'-(1-y)log(1-y') = \begin{cases}-logy',& \text{y=1}\\-log(1-y'),& \text{y=0} \end{cases}
$$
y'是经过激活函数的输出，所以在0-1之间。可见普通的交叉熵对于正样本而言，输出概率越大损失越小。对于负样本而言，输出概率越小则损失越小。此时的损失函数在大量简单样本的迭代过程中比较缓慢且可能无法优化至最优。

Focal loss的改进
$$
L_{fl}= \begin{cases}-(1-y')^\gamma logy',& y=1\\ -y'^\gamma log(1-y'), & y=0 \end{cases}
$$
<img src="C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210625185606891.png" alt="image-20210625185606891" style="zoom: 67%;" />

首先在原有的基础上加了一个因子，其中gamma>0使得减少易分类样本的损失。使得更关注于困难的、错分的样本。

例如gamma为2，对于正类样本而言，预测结果为0.95肯定是简单样本，所以（1-0.95）的gamma次方就会很小，这时损失函数值就变得更小。而预测概率为0.3的样本其损失相对很大。对于负类样本而言同样，预测0.1的结果应当远比预测0.7的样本损失值要小得多。对于预测概率为0.5时，损失只减少了0.25倍，所以更加关注于这种难以区分的样本。这样减少了简单样本的影响，大量预测概率很小的样本叠加起来后的效应才可能比较有效。

此外，加入平衡因子alpha，用来平衡正负样本本身的比例不均：文中alpha取0.25，即正样本要比负样本占比小，这是因为负例易分。
$$
L_{fl}= \begin{cases}-\alpha(1-y')^\gamma logy',& y=1\\ -(1-\alpha)y'^\gamma log(1-y'), & y=0 \end{cases}
$$
只添加alpha虽然可以平衡正负样本的重要性，但是无法解决简单与困难样本的问题。

gamma调节简单样本权重降低的速率，当gamma为0时即为交叉熵损失函数，当gamma增加时，调整因子的影响也在增加。实验发现gamma为2是最优。

#### Kmeans边框大小聚类

```python
import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = "2007_train.txt"

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "2007_train.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
```

### SSD

网络结构

![image-20210712212429956](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210712212429956.png)

![image-20210712212429956](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\20190427212254620.png)

算法步骤：

1、输入一幅图片（300x300），将其输入到预训练好的分类网络中来获得不同大小的特征映射，修改了传统的VGG16网络；

- 将VGG16的FC6和FC7层转化为卷积层，如图1上的Conv6和Conv7；
- 去掉所有的Dropout层和FC8层；
- 添加了Atrous算法（hole算法）；
- 将Pool5从2x2-S2变换到3x3-S1；

2、抽取Conv4_3、Conv7、Conv8_2、Conv9_2、Conv10_2、Conv11_2层的feature map，然后分别在这些feature map层上面的每一个点构造6个不同尺度大小的bbox，然后分别进行检测和分类，生成多个bbox，如图2所示；

3、将不同feature map获得的bbox结合起来，经过NMS（非极大值抑制）方法来抑制掉一部分重叠或者不正确的bbox，生成最终的bbox集合（即检测结果）；
