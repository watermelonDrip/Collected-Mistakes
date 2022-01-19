# 概述
k-近邻采用测量不同特征之间的距离方法进行分类
# Knn 原理
1.  样本数据（训练样本），包含数据和分类的关系
2.  输入没有标签的数据后，将新数据的每个特征与样本集中的数据特征比较
   - 计算距离(新数据和样本集中每一个都算）
   - 距离排序
   - 取前k个样本的数据标签
3. k个样本中，出现最多的标签
# 流程
+ 收集数据：任何方法
+ 准备数据：方便距离计算，想要的数据结构
+ 分析：任何
+ 测试算法：计算错误率
+ 使用算法：输入样本数据和结构化的输出结果（准备数据）
# 优缺点
+ 优点：精度高、对异常值不敏感、无数据输入假定
+ 缺点：计算复杂度高，空间高

# Code

### 伪代码
## 准备：使用python 导入数据
```
from numpy import * # 导入科学计算包numpy
import operator     # 运算符模块
def createdataSet():
    group = array([1,1.1],[1,1],[0,0],[0,0.1])
    labels = ['A','A','B','B']
    return group, labels
```
## 从文本文件中解析数据
```python 
# knn 算法
def classify0(inx,dataSet,labels,k): 
  # 计算距离
    dataSetSize = dataSet.shape[0]      # 训练样本集dataset的行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #也就是每个测试点都和样本集里的元素算一遍距离
    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet的第一个点的距离。
       第二行： 同一个点 到 dataSet的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis = 1)  # 计算的所有点的距离
    distances = sqDistance ** 0.5         # 开方
    sortedDistIndicies = distances.argsort() # 从小到大排序
 # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels(sortedDistIndicies[i])
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
 # 排序
    sortedclassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse = True)
    return sortedclassCount[0][0]
```
输入：inx(用于分类的输入变量), dataSet(训练样本集）,labels(标签向量）,k(选择最近邻居的数目）

 
### 案例1
1. 收集数据: 数据存放在文本.txt中。数据格式是:特征1，特征2，特征3，哪一类人
2. 准备数据：文本转成numpy
```python
def file2matrix(filename): # filename: 数据文件路径
    # 打开文本，得到文件的行数
    fr = open(filename)
    numberOfLines = len(fr.readlines()) # 获取文件数据行行数
    returnMat = zeros((numberOfLines,3)) # 生成空矩阵，为了返回
    classLabelVector = [] # 生成labels向量
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip() # 截取掉所有的回车字符
        listFromLine = line.split('\t') #以'\t'切割字符串
        returnMat[index,:] = listFromLine[0:3] #每列属性
        # 需要明确的通知解释器，存储的元素值为整型，否则python会将元素当作字符串处理
        classLabelVector.append(int(listFromLine[-1]))  
        index+=1
    # 返回数据矩阵returnMat,对应的label向量classLabelVector
    return returnMat,classLabelVector 
```
 
3. 分析数据：使用Matplotlib画二维散点
4. 归一化数值
在处理不同取值范围的特征值时，一般采用数值归一化。`newValue = (oldValue - min)/(max - min) #max最大特征值，min最小特征值
`
```
def autoNorm(dataSet):
    minVals = dataSet.min(0) #每列最小值，参数0表示的是从列中找
    maxVals = dataSet.max(0) #每列最大值
    ranges = maxVals - minVals # 尺寸是1*3 （3个不同特征）
    normDataSet = zeros(shape(dataSet)) 
    m = dataSet.shape[0] 
    normDataSet = dataSet - tile(minVals, (m, 1)) # 把minVals尺寸和dataSet 尺寸一致
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide 
    return normDataSet, ranges, minVals
```
5. 测试算法
使用错误率来检测分类器的性能。
```
def datingClassTest():
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为训练样本
    datingDataMat, datingLabels = file2matrix('input/2.KNN/datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat) # 归一化数据
    m = normMat.shape[0]   # 测试向量的数量（行数，也就是样本数） 
    numTestVecs = int(m * hoRatio) # 哪些用于测试
    errorCount = 0.0
    for i in range(numTestVecs): 
        # 对测试样本进行测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
```
### 案例2
手写识别系统
1. 流程
+ 收集数据：提供文本文件
+ 准备数据：将图像格式转换为分类器使用的list
+ 分析数据
+ 训练算法：不适合knn
+ 测试算法：测试样本，预测样本。测试样本是已经完成分类的数据，如果预测不同，这记错。（一个预测样本跟所有的测试样本比较，而得到结果。所以训练器不用训练）

2. 准备数据： 将图像转换为测试向量(将数据处理成分类器可以识别的格式）
trainingDigits: 2000个例子 （每一个文件是一个数字）
testDigits: 900个例子
使用trainingDigits中来训练分类器，使用testDigits中数据来测试分类器。
    + 将图像数据转换为向量
    + :param filename: 图片文件 因为我们的输入数据的图片格式是 32 * 32的
    + :return: 一维矩阵
    + 该函数将图像转换为向量：该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    + 循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
```python
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect
```
3. 测试算法： 使用knn
将数据输入到分类器
```python
def handwritingClassTest():
    # 1. 导入数据，将trainingDigits目录中的文件内存存储再列表中
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList) #训练数据长度
    trainingMat = zeros((m, 1024)) # 所有的测试样本都存在trainingMat 里，m行1024列
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('input/2.KNN/trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = listdir('input/2.KNN/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('input/2.KNN/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))
```

可以改变k的值，修改函数handwritingClassTest随机选取训练样本、改变训练样本数目，都会对knn错误率有影响

# 总结
![image](https://user-images.githubusercontent.com/69283174/150143860-40074c16-dcb0-4eab-8051-fb7f7e1dc236.png)

 

