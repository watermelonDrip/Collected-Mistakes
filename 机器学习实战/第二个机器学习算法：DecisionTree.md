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
```python
def chooseBestFeatureTopSplit(dataSet):
    numFeatures = len(dataSet[0])-1      # 有多少特征
    baseEntropy = calcShannonEnt(dataSet) # 整个数据集的熵
    bestInfoGain = 0.0; bestFeature = -1   # 最优的信息增益值, 和最优的Featurn编号
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 获取每一个实例的第i+1个feature，组成list集合
        uniqueVals = set(featList)
        newEntropy = 0.0     # 创建一个临时的信息熵
        for value in uniqueVals:   # 遍历某一列的value集合，计算该列的信息熵 
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，
        # 并对所有唯一特征值得到的熵求和
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
```
            

```
chooseBestFeatureToSplit(myDat)
```
```
featList= [1, 1, 1, 0, 0] #选取i=0列，
subDataSet = [[1, 'yes'], [1, 'yes'], [0, 'no']] # i=0特征列表等于1，但是扣掉这一列的subdataset
infoGain= 0.4199730940219749 bestFeature= 0 0.9709505944546686 0.5509775004326937
featList=[1, 1, 0, 1, 1]
subDataSet = [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]
infoGain= 0.17095059445466854 bestFeature= 1 0.9709505944546686 0.8
0
```
          
可以看出，第0个特征是最好的用于划分数据集的特征。

6. 递归构造决策树
第一次划分之后，数据将向下传递。

递归结束的条件： 程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。
+ 如果数据集已经处理了所有属性，但是类标签依然不是唯一的，需要定义叶子节点。一般通常采用多数表决的方法。
```python
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): #存储每个类标签出现的频率
            classCount[vote] = 0
        classCount[vote] +=1
        # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果（注意这里面的输出的不是特征flipper的那个）
        sortedclassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
        return sortedclassCount[0][0]
```
这段代码类似于knn里面的投票表决代码。

```
def createTree(dataSet, labels): #dataSet数据集，labels标签列表
    classList = [example[-1] for example in dataSet] # 第一次迭代的时候是，classlist ['yes', 'yes', 'no', 'no', 'no']
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    if classList.count(classList[0]) == len(classList):    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    if len(dataSet[0]) == 1:     # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
        return majorityCnt(classList)
    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet) #第一迭代 bestFeat = 0
    # 获取label的名称
    bestFeatLabel = labels[bestFeat] # bestFeatLabel: no surfacing
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print 'myTree', value, myTree
    return myTree
```
output:
```
myTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
```
myTree包含了很多代表树结果信息的嵌套字典，从左往右，第一个关键字 'no surfacing'是第一个划分数据集的特征名称，
该关键字的值也是另一个数据字典。第二个关键字是'no surfacing'特征划分的数据集，这些关键字的值是'no surfacing'节点的子节点。
如果值是另一个数据字典比如 ` 1: {'flippers': {0: 'no', 1: 'yes'}}`，子节点上一个判断节点。

### 使用Matplotlib注解绘制树形图
决策树的主要优点就是直观易于理解。 使用Matplotlib库来创建树形图。

```python
import matplotlib.pyplot as plt
#定义文本框喝箭头格式
decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle = "<-")
#绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
def createPlot(inTree):
    
    fig = plt.figure(1, facecolor='green')
    fig.clf()
    #axprops = dict(xticks=[], yticks=[])
    # 表示创建一个1行，1列的图，createPlot.ax1 为第 1 个子图，
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node',  (0.5,0.1), (0.1,0.5), decisionNode)
    plotNode('a leaf node',  (0.8,0.1), (0.3,0.8), leafNode) 
    plt.show()
```

![image](https://user-images.githubusercontent.com/69283174/150350680-32833b93-eebd-41bf-932d-25dd8c958589.png)

### 构造注解树
接下来我们需要知道多少个叶节点，和树有多少层。 叶节点可以确定x轴长度。树层确定y高度。
```python
def getNumLeafs(myTree):
    numLeafs = 0
     
    firstStr = list(myTree.keys())[0] #注意在python3里，dict.keys() returns an iterable but not indexable object. The most simple (but not so efficient) solution would be:`vocab = list(fdist1.keys())`
     
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是否为dict, 不是+1
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是不是dict, 求分枝的深度
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 记录最大的分支深度
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
```













