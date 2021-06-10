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

熵计算公式

![image-20210607145054971](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210607145054971.png)

#### **香农熵**

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

#### **数据集划分**

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

#### **信息增益计算**

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

#### **决策树构建**

```python
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():classCount[vote] = 0   
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        #根据字典的值降序排序
    return sortedClassCount[0][0]                                #返回classList中出现次数最多的元素

def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]            #取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                    #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #选择最优特征
    bestFeatLabel = labels[bestFeat]                            #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                                    #根据最优特征的标签生成树
    del(labels[bestFeat])                                        #删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]        #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                    #遍历特征，创建决策树。                       
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree
```

#### 使用决策树执行分类

```python
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))                                                        #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)                                               
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel
```

#### **决策树的存储**

```python
def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
        
def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
```

### 朴素贝叶斯

#### 贝叶斯推断

条件概率

![image-20210608145111344](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210608145111344.png)

- p(c)称为先验概率 (prior probability), 即在x发生之前，我们对c事件概率的一个判断。
- p(c|x)称为后验概率 (posterior probability), 即在x发生之后，我们对c事件概率的重新评估。
- p(x|c)/p(x)称为可能性函数 (likelihood), 这是一个调整因子，使得预估概率更接近真实概率。

所以，条件概率可以理解为下面的式子：

> 后验概率 = 先验概率 * 调整因子

#### 朴素贝叶斯

朴素贝叶斯对条件概率分布做了条件独立性假设，比如下面的公式，假设有n个特征：

p(a|X) = p(X|a)p(a) = p(x<sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>,...,x<sub>n</sub>|a)p(a)

由于每个特征都是独立的，我们可以进一步拆分公式：

p(a|X) = p(X|a)p(a) = {p(x<sub>1</sub>|a)\*p(x<sub>2</sub>|a)\*p(x<sub>3</sub>|a)\*...\*p(x<sub>n</sub>|a)}p(a)

#### **实例**

某个医院早上来了六个门诊的病人，他们的情况如下表所示：

| 症状   | 职业     | 疾病   |
| ------ | -------- | ------ |
| 打喷嚏 | 护士     | 感冒   |
| 打喷嚏 | 农夫     | 过敏   |
| 头痛   | 建筑工人 | 脑震荡 |
| 头痛   | 建筑工人 | 感冒   |
| 打喷嚏 | 教师     | 感冒   |
| 头痛   | 教师     | 脑震荡 |

现在又来了第七个病人，是一个打喷嚏的建筑工人。请问他患上感冒的概率有多大？

根据贝叶斯定理：
$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$
可得
$$
P(感冒|打喷嚏*建筑工人) = \frac{P(打喷嚏*建筑工人|感冒)P(感冒)}{P(打喷嚏*建筑工人)}
$$
根据朴素贝叶斯条件独立性的假设可知，打喷嚏和建筑工人这两个特征式独立的，因此，上面的等式就变成了
$$
P(感冒|打喷嚏*建筑工人) = \frac{P(打喷嚏|感冒)P(建筑工人|感冒)P(感冒)}{P(打喷嚏)P(建筑工人)}
$$
这里可以计算：
$$
P(感冒|打喷嚏*建筑工人) = \frac{0.66*0.33*0.5}{0.5*0.33} = 0.66
$$

#### 言论过滤器

```python
#建立实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                               #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

#转化为词向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)                                    #创建一个其中所含元素都为0的向量
    for word in inputSet:                                                #遍历每个词条
        if word in vocabList:                                            #如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec                                                    #返回文档向量

#建立词汇表
def createVocabList(dataSet):
    vocabSet = set([])                      #创建一个空的不重复列表
    for document in dataSet:               
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                            #计算训练的文档数目
    numWords = len(trainMatrix[0])                            #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)        #文档属于侮辱类的概率
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)    #创建numpy.zeros数组,词条出现数初始化为0
    p0Denom = 0.0; p1Denom = 0.0                            #分母初始化为0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                            #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                                #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom                                      
    p0Vect = p0Num/p0Denom         
    return p0Vect,p1Vect,pAbusive                            #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

#朴素贝叶斯分类器分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1    			#对应元素相乘
	p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
	print('p0:',p0)
	print('p1:',p1)
	if p1 > p0:
		return 1
	else: 
		return 0
```

#### 朴素贝叶斯改进

- 拉普拉斯平滑

利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，即计算p(w0|1)p(w1|1)p(w2|1)。如果其中有一个概率值为0，那么最后的成绩也为0。

显然，这样是不合理的，为了降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。这种做法就叫做拉普拉斯平滑(Laplace Smoothing)又被称为加1平滑，是比较常用的平滑方法，它就是为了解决0概率问题。

除此之外，另外一个遇到的问题就是下溢出，这是由于太多很小的数相乘造成的。学过数学的人都知道，两个小数相乘，越乘越小，这样就造成了下溢出。在程序中，在相应小数位置进行四舍五入，计算结果可能就变成0了。为了解决这个问题，对乘积结果取自然对数。通过求对数可以避免下溢出或者浮点数舍入导致的错误。同时，采用自然对数进行处理不会有任何损失。下图给出函数f(x)和ln(f(x))的曲线。

**改进的函数**

```python
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                            #计算训练的文档数目
    numWords = len(trainMatrix[0])                            #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)        #文档属于侮辱类的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)    #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0                            #分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:          #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                              #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)                            #取对数，防止下溢出         
    p0Vect = np.log(p0Num/p0Denom)         
    return p0Vect,p1Vect,pAbusive                            #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
```

### Logistic回归

#### 概述

假设现在有一些数据点，我们利用一条直线对这些带你进行拟合（该线称为最佳拟合直线），这个拟合过程就称作回归。Logistic回归是一种二分类算法，他利用的是sigmoid函数阈值在[0,1]这个特性。

sigmoid函数：
$$
\sigma(x) = \frac{1}{1+e^{-z}}
\\
\sigma^{'}(x) = \sigma(x)(1-\sigma(x))
$$
图像

<img src="C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210609142110571.png" alt="image-20210609142110571" style="zoom:80%;" />

根据sigmoid函数的特性，我们可以做出以下的假设
$$
P(y = 1 |x,\theta) = h_\theta(x) 
$$

$$
P(y = 0 |x,\theta) = 1 - h_\theta(x)
$$

上式即为在已知样本x和参数θ的情况下，样本x属性正样本(y=1)和负样本(y=0)的条件概率。理想状态下，根据上述公式，求出各个点的概率均为1，也就是完全分类都正确。但是考虑到实际情况，样本点的概率越接近于1，其分类效果越好。比如一个样本属于正样本的概率为0.51，那么我们就可以说明这个样本属于正样本。另一个样本属于正样本的概率为0.99，那么我们也可以说明这个样本属于正样本。但是显然，第二个样本概率更高，更具说服力。我们可以把上述两个概率公式合二为一：
$$
Cost(h_\theta(x),y) = h_\theta(x)^y(1-h_\theta(x))^{(1-y)}
$$
合并出来的Cost，我们称之为代价函数(Cost Function)。当y等于1时，(1-y)项(第二项)为0；当y等于0时，y项(第一项)为0。为了简化问题，我们对整个表达式求对数，(将指数问题对数化是处理数学问题常见的方法)：
$$
Cost(h_\theta(x),y) = y\log h_\theta(x)+(1-y)\log(1-h_\theta(x))
$$
这个代价函数，是对于一个样本而言的。给定一个样本，我们就可以通过这个代价函数求出，样本所属类别的概率，而这个概率越大越好，所以也就是求解这个代价函数的最大值。既然概率出来了，那么最大似然估计也该出场了。假定样本与样本之间相互独立，那么整个样本集生成的概率即为所有样本生成概率的乘积，再将公式对数化，便可得到如下公式：
$$
J(\theta) = \sum_{i=1}^{m}[y^{i}\log h_\theta(x^{i})+(1-y^{i})\log(1-h_\theta(x^{i}))]
$$
其中，m为样本的总数，y(i)表示第i个样本的类别，x(i)表示第i个样本，需要注意的是θ是多维向量，x(i)也是多维向量。

**综上所述，满足J(θ)的最大的θ值即是我们需要求解的模型。**

求解函数的极大值使用梯度上升算法，函数添加负号，即变为梯度下降算法来求极小值。

通用求解公式：
$$
\theta _j := \theta _j + \alpha\frac{\partial J(\theta)}{\theta _j}
$$

#### 梯度上升

```python
# -*- coding:UTF-8 -*-
import numpy as np

"""
函数说明:加载数据

Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
"""
def loadDataSet():
    dataMat = []                                                        #创建数据列表
    labelMat = []                                                        #创建标签列表
    fr = open('testSet.txt')                                            #打开文件   
    for line in fr.readlines():                                            #逐行读取
        lineArr = line.strip().split()                                    #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])        #添加数据
        labelMat.append(int(lineArr[2]))                                #添加标签
    fr.close()                                                            #关闭文件
    return dataMat, labelMat                                            #返回

"""
函数说明:sigmoid函数

Parameters:
    inX - 数据
Returns:
    sigmoid函数
"""
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
函数说明:梯度上升算法

Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                      #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                    #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                  #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                                                #将矩阵转换为数组，返回权重数组
```

#### 改进的随机梯度上升算法

梯度上升算法在每次更新回归系数(最优参数)时，都需要遍历整个数据集。该方法处理100个左右的数据集时尚可，但如果有数十亿样本和成千上万的特征，那么该方法的计算复杂度就太高了。因此，需要对算法进行改进，我们每次更新回归系数(最优参数)的时候，能不能不用所有样本呢？一次只用一个样本点去更新回归系数(最优参数)？这样就可以有效减少计算量了，这种方法就叫做随机梯度上升算法。

```python
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)                                       #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                             #参数初始化
    for j in range(numIter):                                           
        dataIndex = list(range(m))
        for i in range(m):           
            alpha = 4/(1.0+j+i)+0.01                                     #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       #更新回归系数
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    return weights                                                      #返回
```

#### 使用 Sklearn 构建 Logistic 回归分类器

![image-20210609151454818](C:\Users\jwliu\AppData\Roaming\Typora\typora-user-images\image-20210609151454818.png)

优化算法

- solver：优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear。solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是：
  - liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
  - lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
  - newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
  - sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
  - saga：线性收敛的随机优化算法的的变重。
  - 总结：
    - liblinear适用于小数据集，而sag和saga适用于大数据集因为速度更快。
    - 对于多分类问题，只有newton-cg,sag,saga和lbfgs能够处理多项损失，而liblinear受限于一对剩余(OvR)。啥意思，就是用liblinear的时候，如果是多分类问题，得先把一种类别作为一个类别，剩余的所有类别作为另外一个类别。一次类推，遍历所有类别，进行分类。
    - newton-cg,sag和lbfgs这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear和saga通吃L1正则化和L2正则化。
    - 同时，sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，比如大于10万，sag是第一选择。但是sag不能用于L1正则化，所以当你有大量的样本，又需要L1正则化的话就要自己做取舍了。要么通过对样本采样来降低样本量，要么回到L2正则化。
    - 从上面的描述，大家可能觉得，既然newton-cg, lbfgs和sag这么多限制，如果不是大样本，我们选择liblinear不就行了嘛！错，因为liblinear也有自己的弱点！我们知道，逻辑回归有二元逻辑回归和多元逻辑回归。对于多元逻辑回归常见的有one-vs-rest(OvR)和many-vs-many(MvM)两种。而MvM一般比OvR分类相对准确一些。郁闷的是liblinear只支持OvR，不支持MvM，这样如果我们需要相对精确的多元逻辑回归时，就不能选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归不能使用L1正则化了。
      
