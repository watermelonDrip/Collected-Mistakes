# 基于数据集多重抽样的分类器

将不同的分类器组合起来，称为集成方法或元算法

## bagging 和 boosting
目前 bagging 方法最流行的版本是: 随机森林(random forest)
选男友: 美女选择择偶对象的时候，会问几个闺蜜的建议，最后选择一个综合得分最高的一个作为男朋友

目前 boosting 方法最流行的版本是: AdaBoost
追女友: 3个帅哥追同一个美女，第1个帅哥失败->(传授经验: 姓名、家庭情况) 第2个帅哥失败->(传授经验: 兴趣爱好、性格特点) 第3个帅哥成功

bagging 和 boosting 区别是什么？

+ bagging 是一种与 boosting 很类似的技术, 所使用的多个分类器的类型（数据量和特征量）都是一致的。
+ bagging 是由不同的分类器（1.数据随机化 2.特征随机化）经过训练，综合得出的出现最多分类结果；boosting 是通过调整已有分类器错分的那些数据来获得新的分类器，得出目前最优的结果。
+ bagging 中的分类器权重是相等的；而 boosting 中的分类器加权求和，所以权重并不相等，每个权重代表的是其对应分类器在上一轮迭代中的成功度。

## Adaboost 开发流程

收集数据: 可以使用任意方法
准备数据: 依赖于所使用的弱分类器类型，本章使用的是单层决策树，这种分类器可以处理任何数据类型。
    当然也可以使用任意分类器作为弱分类器，第2章到第6章中的任一分类器都可以充当弱分类器。
    作为弱分类器，简单分类器的效果更好。
分析数据: 可以使用任意方法。
训练算法: AdaBoost 的大部分时间都用在训练上，分类器将多次在同一数据集上训练弱分类器。
测试算法: 计算分类的错误率。
使用算法: 通SVM一样，AdaBoost 预测两个类别中的一个。如果想把它应用到多个类别的场景，那么就要像多类 SVM 中的做法一样对 AdaBoost 进行修改。


## 训练算法：基于错误提升分类器的性能

AdaBoost是adaptive boosting的缩写，运行过程如下： 训练数据中的每个样本，并赋予其一个权重，这些权重构成了向量D。
一开始，这些权重都初始化成相等值。首先在训练数据上训练出一个弱分类器并计算该分类器的错误率，然后在同一个数据集上再次训练弱分类器。
在分类器的第二次训练当中，将会重新调整每个样本的权重，对第一次分对的样本的权重护降低，而分错的提高。

# 基于单层决策树构建弱分类器

## 实现数据集和类标签的导入

```python
def loadSimpData():
    """ 测试数据
    Returns:
        dataArr   feature对应的数据集
        labelArr  feature对应的分类标签
    """
    dataArr = array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    labelArr = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataArr, labelArr
 ```
 ## 测试是否有某个值小于或大于我们正在测试的阈值
 
 ```python
 def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    """stumpClassify(将数据集，按照feature列的value进行 二分法切分比较来赋值分类)
    Args:
        dataMat    Matrix数据集
        dimen      特征列
        threshVal  特征列要比较的值
    Returns:
        retArray 结果集
    """
    # 默认都是1
    retArray = ones((shape(dataMat)[0], 1))
    # dataMat[:, dimen] 表示数据集中第dimen列的所有值
    # threshIneq == 'lt'表示修改左边的值，gt表示修改右边的值
    if threshIneq == 'lt':
        retArray[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1.0
    return retArray
```

## 找到具有最低错误率的单层决策树

伪代码：
```
将最小错误率minError设置为+infty
对数据集中的每一个特征（第一层循环）：
    对每个步长（第二次循环）：
        对每个不等号（第三层循环）
            建立一棵单层决策树并利用加权数据集对它进行测试
            如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
返回最佳单层决策树
```

```python
#将会遍历stumpClassify()函数的所有可能值，找到最佳的单层决策树。 最佳是基于权值向量D来定义的。

def buildStump(dataArr, labelArr, D):
    """buildStump(得到决策树的模型)

    Args:
        dataArr   特征标签集合
        labelArr  分类标签集合
        D         最初的特征权重值
    Returns:
        bestStump    最优的分类器模型
        minError     错误率
        bestClasEst  训练后的结果集
    """
    # 转换数据
    dataMat = mat(dataArr), labelMat = mat(labelArr).T
    # m行 n列
    m, n = shape(dataMat)
    # 初始化数据
    numSteps = 10.0, bestStump = {}, bestClasEst = mat(zeros((m, 1)))
    # 初始化的最小误差为无穷大
    minError = inf

    # 循环所有的feature列，将列切分成 若干份，每一段以最左边的点作为分类节点
    for i in range(n): # 循环在数据集的所有特征
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        # 计算每一份的元素个数
        stepSize = (rangeMax-rangeMin)/numSteps
        # 例如： 4=(10-1)/2   那么  1-4(-1次)   1(0次)  1+1*4(1次)   1+2*4(2次)
        # 所以： 循环 -1/0/1/2
        for j in range(-1, int(numSteps)+1):
            # go over less than and greater than
            for inequal in ['lt', 'gt']: # 大于小于切换不等式
                # 如果是-1，那么得到rangeMin-stepSize; 如果是numSteps，那么得到rangeMax
                threshVal = (rangeMin + float(j) * stepSize)
                # 对单层决策树进行简单分类，得到预测的分类值
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                # 正确为0，错误为1
                errArr[predictedVals == labelMat] = 0
                # 计算 平均每个特征的概率0.2*错误概率的总和为多少，就知道错误率多高
                # 例如： 一个都没错，那么错误率= 0.2*0=0 ， 5个都错，那么错误率= 0.2*5=1， 只错3个，那么错误率= 0.2*3=0.6
                weightedError = D.T*errArr
                '''
                dim            表示 feature列
                threshVal      表示树的分界值
                inequal        表示计算树左右颠倒的错误率的情况
                weightedError  表示整体结果的错误率
                bestClasEst    预测的最优结果
                '''
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    # bestStump 表示分类器的结果，在第几个列上，用大于／小于比较，阈值是多少
    return bestStump, minError, bestClasEst
```

output:

```
D = mat(ones((5,1))/5)
buildStump(datMat, classLabels, D)
>>> 
split: dim 0, thresh 0.90, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 0.90, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.20, thresh ineqal: lt, the weighted error is 0.400
split: dim 0, thresh 1.20, thresh ineqal: gt, the weighted error is 0.600
split: dim 0, thresh 1.30, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.30, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.40, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.40, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.50, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.50, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.60, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.60, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.70, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.70, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.80, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.80, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 1.90, thresh ineqal: lt, the weighted error is 0.200
split: dim 0, thresh 1.90, thresh ineqal: gt, the weighted error is 0.800
split: dim 0, thresh 2.00, thresh ineqal: lt, the weighted error is 0.600
split: dim 0, thresh 2.00, thresh ineqal: gt, the weighted error is 0.400
split: dim 1, thresh 0.89, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 0.89, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.00, thresh ineqal: lt, the weighted error is 0.200
split: dim 1, thresh 1.00, thresh ineqal: gt, the weighted error is 0.800
split: dim 1, thresh 1.11, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.11, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.22, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.22, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.33, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.33, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.44, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.44, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.55, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.55, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.66, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.66, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.77, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.77, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.88, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.88, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 1.99, thresh ineqal: lt, the weighted error is 0.400
split: dim 1, thresh 1.99, thresh ineqal: gt, the weighted error is 0.600
split: dim 1, thresh 2.10, thresh ineqal: lt, the weighted error is 0.600
split: dim 1, thresh 2.10, thresh ineqal: gt, the weighted error is 0.400
({'dim': 0, 'ineq': 'lt', 'thresh': 1.3}, matrix([[0.2]]), array([[-1.],
        [ 1.],
        [-1.],
        [-1.],
        [ 1.]]))
```
 
上述单层决策树的生成树是决策树的一个简化版本。它就是所谓的弱学习器，也就是弱分类算法。

# 完整的AdaBoost算法

上面我们构建了一个基于加权输入值进行决策的分类器。 
伪代码：
```python
对每次迭代：
    利用buildStump()函数找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树组
    计算alpha
    计算新的权重向量量D
    更新累计类别估计值
    如果错误率等于0.0,退出
```

```python
def adaBoostTrainDS(dataArr, labelArr, numIt=40):
    """adaBoostTrainDS(adaBoost训练过程放大)
    Args:
        dataArr   特征标签集合
        labelArr  分类标签集合
        numIt     实例数
    Returns:
        weakClassArr  弱分类器的集合
        aggClassEst   预测的分类结果值
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化 D，设置每个样本的权重值，平均分为m份
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # 得到决策树的模型
        bestStump, error, classEst = buildStump(dataArr, labelArr, D)

        # alpha目的主要是计算每一个分类器实例的权重(组合就是分类结果)
        # 计算每个分类器的alpha权重值
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        # store Stump Params in Array
        weakClassArr.append(bestStump)

        print "alpha=%s, classEst=%s, bestStump=%s, error=%s " % (alpha, classEst.T, bestStump, error)
        # 分类正确: 乘积为1，不会影响结果，-1主要是下面求e的-alpha次方
        # 分类错误: 乘积为 -1，结果会受影响，所以也乘以 -1
        expon = multiply(-1*alpha*mat(labelArr).T, classEst)
        print '(-1取反)预测值expon=', expon.T
        # 计算e的expon次方，然后计算得到一个综合的概率的值
        # 结果发现:  判断错误的样本，D中相对应的样本权重值会变大。
        D = multiply(D, exp(expon))
        D = D/D.sum()

        # 预测的分类结果值，在上一轮结果的基础上，进行加和操作
        print '当前的分类结果: ', alpha*classEst.T
        aggClassEst += alpha*classEst
        print "叠加后的分类结果aggClassEst: ", aggClassEst.T
        # sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
        # 结果为: 错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
        aggErrors = multiply(sign(aggClassEst) != mat(labelArr).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        # print "total error=%s " % (errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst
```



















 
 
 
