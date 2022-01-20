# 概述
决策树是最经常使用的数据挖掘算法。优势在于数据形式很容易理解。

## 决策树的构造
+ 优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据
+ 缺点：可能会产生过度匹配问题

### 哪个特征在划分数据分类时起决定性作用
为了找到决定性的特征，先评估每个特征。完成测试之后，原始数据集就划分为几个数据子集。 
这些数据子集分布在第一个决策点点所有分支上。如果某个分支下的数据属于同一个类型，无需进一步对数据集进行分割。
如果数据子集内的数据不属于同一类型，这需要重复划分数据子集的过程。

创建分支的伪代码函数createBranch()如下：
```
#检测数据集中的每个子项是否属于同一分类
I so return 类标签：
Else
    寻找划分数据集的最好特征
    划分数据集
    创建分支节点
        for 每个划分的子集
            调用函数createBranc并增加返回结果到分支节点中 #递归
    return 分支节点
```

### 算法是如何划分数据集的
1. 决策树的一般流程

+ 收集数据: 可以使用任何方法
+ 准备数据: 树构造算法（这里使用的是ID3算法，因此数值型数据必须离散化。）
+ 分析数据: 可以使用任何方法，构造树完成之后，我们可以将树画出来。
+ 训练算法: 构造树结构
+ 测试算法: 使用习得的决策树执行分类
+ 使用算法: 此步骤可以适用于任何监督学习任务，而使用决策树可以更好地理解数据的内在含义


2. 如何选择哪个特征作为划分的参考属性

划分数据集最大的原则是：将无序的数据变得更有序。我的理解是，决策就是找规律，所以要更有序。
+ 信息增益：划分数据集之前之后信息发生的变化，获得信息增益最高的就是最好的选择
+ 熵：信息的期望值

3. 计算给定数据集的熵（固定的）

```python
from math import log
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) 
    labelCounts = {}
  # 为所有可能分类创建字典
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]= 0
            labelCounts[currentLabel]+=1
        shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt # 统计所有类标签发生的次数
```

比如：下面是鱼鉴定数据集，`calcShannonEnt(dataSet)`函数里的 'feaVect = [1, 1, 'yes']‘等等。`key = yes` or 'key = no'。 也就是数据的标签。


```python
def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
```
熵越高，混合的数据越多。如果变成 ‘yes','no','maybe'则熵变高。

4. 划分数据集
将对每个特征划分数据集的结果计算一次信息熵，然后判断按照哪个特征划分数据集是最好的划分方式。      
当我们按照某个特征划分数据集时，需要将所有符合要求的元素抽取出来。
```python
# dataSet带划分数据集
# index 划分数据集的特征，选取myDat的那一列
# value 特征返回值
def splitDataSet(dataSet,index,value):
    retDataSet = []
    for featVect in dataSet:  
        if featVect[index] == value: # 判断index列的值是否为value, index列表示的是哪个特征，'no surfacing'还是 'flippers'。
            reducedFeatVec = featVect[:index]  # chop out index used for splitting
            reducedFeatVec.extend(featVect[index+1,:])
            # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
            # 收集结果值 index列为value的行【该行需要排除index列】
            retDataSet.append(reducedFeatVec)
    return retDataSet
```
5. 找到最好的特征划分方式
接下来将遍历整个数据集，循环计算香农熵 和 `splitDataSet()`函数，找到最好的特征划分方式。
熵计算会告诉我们如何划分数据集时最好的数据组织方式。

            
















