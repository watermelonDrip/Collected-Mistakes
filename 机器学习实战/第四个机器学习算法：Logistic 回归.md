首次接触到最优算法

# 概述

# 流程
+ 收集数据：采用任意方法收集数据
+ 准备数据：由于需要进行距离计算，要求数据啥数值型。
+ 分析数据：数据进行分析
+ 训练算法：大部分时间用于训练，找到最佳的分类回归系数
+ 测试算法：一旦训练步骤完成，分类很快
+ 使用算法：

# 基于Logistic回归和Sigmoid 函数分类

+ 优点：计算代码不高，易于理解和实现
+ 缺点：容易欠拟合，分类精度可能不高

  Sigmoid 函数具体的计算公式如下:

![image](https://user-images.githubusercontent.com/69283174/150942159-711c3aa5-4916-4f33-a229-e4896403f96a.png)

![image](https://user-images.githubusercontent.com/69283174/150942285-b2ebe998-c4c3-48d8-90a5-377029477650.png)

为了实现logistic回归分类，可以在每个特征上乘以一个回归系数，然后把所有的结果值相加，总和带入Sigmoid()中，进而
得到一个范围0-1之间的数值。

# 基于最优化方法的最佳回归系数确定

sigmoid 函数的输入z, z= w0x0+w1x1+...+wnxn = W^Tx. 其中向量x是分类器的输入数据，向量w是最佳参数。 为了寻找该最佳参数，
需要用到最优化理论的知识。

## 梯度上升法 
第一个最优化算法叫梯度上升法。基本思想是：要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻。梯度下降是求最小值的。公式一样，正负号不同而已。


梯度算子总是指向函数值增长最快的方向。这里所说的是移动方向，而未提到移动量的大小。该量值称为步长，记作 α 。用向量来表示的话，梯度上升算法的迭代公式如下:

梯度上升迭代公式：![image](https://user-images.githubusercontent.com/69283174/150945513-72ba66cb-5804-41b5-acb8-11bcf8db3d3b.png)


## 训练算法：使用梯度上升找最佳参数
梯度上升伪代码：
```
每个回归系数初始化为1
重复R次：
    计算整个数据集的梯度
    使用alpha ✖️ gradient 更新回归系数的向量
    返回回归系数
```

代码如下：
```
def loadDataSet(): #打开文本文件，并逐行读取。前两个值x1,x2,第三个是对应类别标签。x0=1
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[0])])
        label.append(int(lineArr[2]))
    return dataMat, labelMat
def sigmoid(inx):
    return 1.0/(1+exp(-inx))
def gradAscent(dataMatIn, classLabels):
    '''
    Desc:
        正常的梯度上升法
    Args:
        dataMatIn -- 输入的 数据的特征 List
        classLabels -- 输入的数据的类别标签
    Returns:
        array(weights) -- 得到的最佳回归系数
    '''

    # 转化为矩阵[[1,1,2],[1,1,2]....]
    dataMatrix = array(dataMatIn)             # 转换为 NumPy 矩阵
    labelMat = array(classLabels).transpose() # 首先将数组转换为 NumPy 矩阵，然后再将行向量转置为列向量
    # m->数据量，样本数 n->特征数
    m,n = shape(dataMatrix)
    alpha = 0.001  # alpha代表向目标移动的步长
    maxCycles = 500  # 迭代次数
    # 生成一个长度和特征数相同的矩阵，此处n为3 -> [[1],[1],[1]]
    # weights 代表回归系数， 此处的 ones((n,1)) 创建一个长度和特征数相同的矩阵，其中的数全部都是 1
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(np.matmul(dataMatrix, weights) )     # 矩阵乘法
        # labelMat是实际值
        labelMat =  reshape((labelMat),(m, 1))  
        print(h.shape,labelMat.shape)
        error = np.subtract(labelMat, h)              # 向量相减
        # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，最后得出 x1,x2,xn的系数的偏移量
        tmp = np.matmul(dataMatrix.transpose(), error)
        t = alpha*tmp
        weights = weights +  (alpha*tmp)
    return array(weights)
```

output:

```
dataArr,labelMat = loadDataSet('TestSet.txt')
gradAscent(dataArr, labelMat)
```
## 分析数据：画出决策边界
上面已经解出了一组回归系数，它确定了不同类别数据之间的分割线。怎么画出来便于看呢？
```
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2]); 
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2]); 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    """
    y的由来，卧槽，是不是没看懂？
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值， 因为sigmoid 函数中间的分界点是 0
    g(z>0)>0.5 第一类， g(z<0)<0.5第二类，所以z=0 是分界点。
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X'); plt.ylabel('Y')
    plt.show()
```

    
## 训练算法：随机梯度上升

梯度上升算法在每次更新回归系数时，需要遍历整个数据集，复杂度太高。一种改进方法是一次仅用一个样本点来更新系数，该方法称为随机梯度上升算法。

随机梯度上升伪代码：
```
每个回归系数初始化为1
对数据集中每个样本
    计算样本的梯度
    使用alpha ✖️ gradient 更新回归系数的向量
返回回归系数
```

代码如下：
```python
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights) )   # 矩阵乘法
        error =  classLabels[i] - h               # 向量相减
    
        weights = weights +  (alpha*dataMatrix[i]*error)
    return array(weights)
```

output:
```
w = stocGradAscent0(array(dataArr), labelMat)
plotBestFit(w)
```

![image](https://user-images.githubusercontent.com/69283174/150992416-21a781a7-c594-4a48-8acb-39a197103629.png)

改进随机梯度上升算法

```python
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    Desc:
        改进版的随机梯度上升，使用随机的一个样本来更新回归系数
    Args:
        dataMatrix -- 输入数据的数据特征（除去最后一列数据）
        classLabels -- 输入数据的类别标签（最后一列数据）
        numIter=150 --  迭代次数
    Returns:
        weights -- 得到的最佳回归系数
    '''
    m,n = shape(dataMatrix)
    weights = ones(n)   # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
    # 随机梯度, 循环150,观察是否收敛
    for j in range(numIter):
        # [0, 1, 2 .. m-1]
        dataIndex = [ i for i in range(m)]
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4/(1.0+j+i)+0.0001    # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(random.uniform(0,len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            # print weights, '__h=%s' % h, '__'*20, alpha, '__'*20, error, '__'*20, dataMatrix[randIndex]
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
 ```
 ![image](https://user-images.githubusercontent.com/69283174/150994849-417420c5-acb9-4277-bf26-1f60ba48a566.png)



## 案例
使用Logistic回归预测患有疝病的马存活问题。这里的数据包括368个样本和28个特征。

### 流程
（1）收集数据：给定数据文件
（2）准备数据：用python解析文本文件并填充缺失值
（3）分析数据：可视化并观察数据
（4）训练算法：使用优化算法，找到最佳系数
（5）测试算法：为了量化回归效果，需要观察错误率。 根据错误率决定是否退到训练阶段，通过改变迭代次数和步长等得到最好的回归系数
（6）使用算法

1. 准备数据：处理数据中的缺失值

下面给出了一些可选的做法:

+ 使用可用特征的均值来填补缺失值；
+ 使用特殊值来填补缺失值，如 -1；
+ 忽略有缺失值的样本；
+ 使用有相似样本的均值添补缺失值；
+ 使用另外的机器学习算法预测缺失值

2. 测试算法：用Logistic回归进行分类

用Logistic回归方法进行分类并不需要很多工作，需要做的只是把测试集上的每个特征向量乘以最优化方法得来的回归系数，
再将该乘积结果求和，最后输入到Sigmoid函数中即可。 对应的Sigmoid结果大于0.5，判断1，否则0.

```python

def classifyVector(inX,weights): # 以回归向量和特征
    prob = sigmoid(sum(inX*weights))
    if prob>0.5: return 1.0
    else: return 0.0
    
def colicTest(): 
    frTrain = open('input/5.Logistic/horseColicTraining.txt') # 打开测试集
    frTest = open('input/5.Logistic/horseColicTest.txt')      # 打开训练集
    trainingSet, trainingLabels = [],[]
    # 解析训练数据中的数据特征和label
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): #前21个数据是特征 
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) #第22个是标签
    # 使用 改进后的 随机梯度下降算法 求得在此数据集上的最佳回归系数 trainWeights
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount,numTestVec = 0, 0.0
    # 读取测试数据集进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print （"the error rate of this test is", errorRate）
    return errorRate
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
```
























