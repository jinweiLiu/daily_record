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
