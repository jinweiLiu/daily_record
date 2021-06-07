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
- 测试算法，使用海伦提供的部分数据作为测试样本（测试样本和非测试样本的区别在于，测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误）

#### 手写识别系统

根据给定宽高为32*32的黑白图像，判断图像类别，**将二维图像转化为一维数组进行判断**。

使用k-近邻算法的手写识别系统：

- 收集数据，提供文本文件
- 准备数据，将图像格式转换为分类器使用的list格式
- 分析数据，检查数据，确保符合要求
- 测试算法，编写函数使用提供的部分数据集作为测试样本

代码实现：

```python
# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果
"""
def classify0(inX, dataSet, labels, k):
	#numpy函数shape[0]返回dataSet的行数
	dataSetSize = dataSet.shape[0]
	#在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	#二维特征相减后平方
	sqDiffMat = diffMat**2
	#sum()所有元素相加,sum(0)列相加,sum(1)行相加
	sqDistances = sqDiffMat.sum(axis=1)
	#开方,计算出距离
	distances = sqDistances**0.5
	#返回distances中元素从小到大排序后的索引值
	sortedDistIndices = distances.argsort()
	#定一个记录类别次数的字典
	classCount = {}
	for i in range(k):
		#取出前k个元素的类别
		voteIlabel = labels[sortedDistIndices[i]]
		#dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		#计算类别次数
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#python3中用items()替换python2中的iteritems()
	#key=operator.itemgetter(1)根据字典的值进行排序
	#key=operator.itemgetter(0)根据字典的键进行排序
	#reverse降序排序字典
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#返回次数最多的类别,即所要分类的类别
	return sortedClassCount[0][0]

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
	filename - 文件名
Returns:
	returnVect - 返回的二进制图像的1x1024向量
"""
def img2vector(filename):
	#创建1x1024零向量
	returnVect = np.zeros((1, 1024))
	#打开文件
	fr = open(filename)
	#按行读取
	for i in range(32):
		#读一行数据
		lineStr = fr.readline()
		#每一行的前32个元素依次添加到returnVect中
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	#返回转换后的1x1024向量
	return returnVect

"""
函数说明:手写数字分类测试

Parameters:
	无
Returns:
	无
"""
def handwritingClassTest():
	#测试集的Labels
	hwLabels = []
	#返回trainingDigits目录下的文件名
	trainingFileList = listdir('trainingDigits')
	#返回文件夹下文件的个数
	m = len(trainingFileList)
	#初始化训练的Mat矩阵,测试集
	trainingMat = np.zeros((m, 1024))
	#从文件名中解析出训练集的类别
	for i in range(m):
		#获得文件的名字
		fileNameStr = trainingFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#将获得的类别添加到hwLabels中
		hwLabels.append(classNumber)
		#将每一个文件的1x1024数据存储到trainingMat矩阵中
		trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
	#返回testDigits目录下的文件名
	testFileList = listdir('testDigits')
	#错误检测计数
	errorCount = 0.0
	#测试数据的数量
	mTest = len(testFileList)
	#从文件中解析出测试集的类别并进行分类测试
	for i in range(mTest):
		#获得文件的名字
		fileNameStr = testFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#获得测试集的1x1024向量,用于训练
		vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
		#获得预测结果
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest))


"""
函数说明:main函数

Parameters:
	无
Returns:
	无
"""
if __name__ == '__main__':
	handwritingClassTest()
```

还可以使用 **sklearn.neighbors** 中 **KNeighborsClassifier** 的进行测试。



### 决策树

决策树的建立如下图邮件分类过程，简单的来看是if else的组合，根据特征对数据集进行划分。

![image-20210607110317848](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210607110317848.png)

如何对数据集进行划分（即特征选择的次序），这时就需要信息增益。

**香农熵**

```python
def calcShannonEnt(dataSet):
	numEntires = len(dataSet)						#返回数据集的行数
	labelCounts = {}								#保存每个标签(Label)出现次数的字典
	for featVec in dataSet:							#对每组特征向量进行统计
		currentLabel = featVec[-1]					#提取标签(Label)信息
		if currentLabel not in labelCounts.keys():	#如果标签(Label)没有放入统计次数的字典,添加进去
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1				#Label计数
	shannonEnt = 0.0								#经验熵(香农熵)
	for key in labelCounts:							#计算香农熵
		prob = float(labelCounts[key]) / numEntires	#选择该标签(Label)的概率
		shannonEnt -= prob * log(prob, 2)			#利用公式计算
	return shannonEnt								#返回经验熵(香农熵)
#熵越大，表明数据越混乱
```

**数据集划分**

```python
def splitDataSet(dataSet, axis, value):		
	retDataSet = []										#创建返回的数据集列表
	for featVec in dataSet: 							#遍历数据集
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]				#去掉axis特征
			reducedFeatVec.extend(featVec[axis+1:]) 	#将符合条件的添加到返回的数据集
			retDataSet.append(reducedFeatVec)
	return retDataSet		  							#返回划分后的数据集
```

**信息增益计算**

```python
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1					#特征数量
	baseEntropy = calcShannonEnt(dataSet) 				#计算数据集的香农熵
	bestInfoGain = 0.0  								#信息增益
	bestFeature = -1									#最优特征的索引值
	for i in range(numFeatures): 						#遍历所有特征
		#获取dataSet的第i个所有特征
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)     					#创建set集合{},元素不可重复
		newEntropy = 0.0  								#经验条件熵
		for value in uniqueVals: 						#计算信息增益
			subDataSet = splitDataSet(dataSet, i, value) 		#subDataSet划分后的子集
			#print(subDataSet)
			prob = len(subDataSet) / float(len(dataSet))   		#计算子集的概率
			newEntropy += prob * calcShannonEnt(subDataSet) 	#根据公式计算经验条件熵
		infoGain = baseEntropy - newEntropy 					#信息增益
		# print("第%d个特征的增益为%.3f" % (i, infoGain))			#打印每个特征的信息增益
		if (infoGain > bestInfoGain): 							#计算信息增益
			bestInfoGain = infoGain 							#更新信息增益，找到最大的信息增益
			bestFeature = i 									#记录信息增益最大的特征的索引值
	return bestFeature 											#返回信息增益最大的特征的索引值
```

