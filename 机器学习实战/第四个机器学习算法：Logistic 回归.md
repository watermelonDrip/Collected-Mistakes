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









