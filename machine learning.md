### K-近邻算法

​        k-近邻算法采用测量不同特征值之间的距离方法进行分类。

​        工作原理：存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中k的出处，通常是不大于20的整数。最后选择k个最相似数据中出现次数最多的分类，作为新数据的分类。

#### 电影类别判断

​        根据电影的两个特征（打斗镜头、接吻镜头）来判断电影的类型（爱情片，动作片）。算法步骤如下：

- 计算已知类别数据集中的点与当前点之间的距离；
- 按照距离递增次序排序
- 选取与当前点距离最小的k个点
- 确定前k个点所在类别的出现概率
- 返回前k个点出现频率最高的类别作为当前点的预测分类

代码实现：

```python
def createDataSet():
    #特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels

'''
Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果
'''
def classify(inx,dataset,labels,k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataset.shape[0]
    #扩充inx，再计算差值
    diffMat = np.tile(inx,(dataSetSize,1)) - dataset
    #二维特征相减后平方
    sqDiffMat = diffMat ** 2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，计算距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        votelable = labels[sortedDistIndices[i]]
        classCount[votelable] = classCount.get(votelable,0)+1
    #operator模块提供的itemgetter函数主要用于获取某一对象 特定维度的数据
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) 
    return sortedClassCount[0][0]

if __name__ == '__main__':
    #创建数据集
    group,labels = createDataSet()
    #测试集
    test = [5,20]
    test_class = classify(test,group,labels,3)
    print(test_class)
```

#### 海伦约会

根据男生的特征（每年获得的飞行常客里程数，玩视频游戏所耗时间百分比，每周消费的冰淇淋公升数）判断海伦对于男生的喜好程度（不喜欢的人，魅力一般的人，极具魅力的人）

**不同特征的值分布差距可能比较大，需要进行归一化处理，new_Value = (old_Value - min) / (max - min)**

k-邻近算法步骤：

- 收集数据，提供文本文件
- 准备数据，使用python解析文本文件
- 分析数据，使用matplotlib画二维扩散图
- 测试算法，使用海伦提供的部分数据作为测试样本（测试样本和非测试样本的区别在于，测试样本是已经完成分类的数据，如果）