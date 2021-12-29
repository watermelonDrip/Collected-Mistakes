## 2. TensorFlow 中的线性回归

在本章中，我将开始使用简单模型：线性回归来探索 pytorch 编程。基于这个例子，我将介绍一些代码基础知识，以及，如何调用学习过程中的各种重要组件，如函数函数或算法梯度下降。

### 变量之间的关系模型

线性回归是一种用于衡量变量之间关系的统计技术。它的有趣之处在于实现它的算法在概念上不复杂，并且还可以适应各种各样的情况。由于这些原因，我发现用线性回归的例子开始深入研究 TensorFlow 很有意思。

请记住，在两个变量（简单回归）和两个以上变量（多元回归）的情况下，线性回归拟合因变量和自变量之间的关系`xi`和随机项`b`。

在本节中，我将创建一个简单的示例来解释 pytorch 如何工作，假设我们的数据模型对应简单的线性回归`y = W * x + b`。为此，我使用一个简单的 Python 程序在二维空间中创建数据，然后我会要求 TensorFlow 在这些点上寻找最适合的直线。

首先要做的是导入我们将用于生成点的 NumPy 包。我们创建的代码如下：

```py
import numpy as np
 
num_points = 1000
vectors_set = []
for i in range(num_points):
         x1= np.random.normal(0.0, 0.55)
         y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])
 
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
```

从代码中可以看出，我们根据关系`y = 0.1 * x + 0.3`生成了点，尽管有一些正态分布的变化，因此这些点并不完全对应一条线，让我们编写一个更有趣的例子。

在我们的例子中，所得到的点云是：

![image](https://user-images.githubusercontent.com/69283174/147643137-9a890fce-9614-4fd2-8e4c-4f25a493a198.png)

读者可以使用以下代码查看它们（这里，我们需要导入`matplotlib`包的一些函数，运行`pip install matplotlib` [13]）：

```py
import matplotlib.pyplot as plt
 
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()
```

这些点是我们将考虑的模型的训练数据集的数据。

### 损失函数和梯度下降算法

下一步是训练我们的学习算法，以便能够获得从输入数据`x_data`估计的输出值`y`。在这种情况下，正如我们事先所知，它是线性回归，我们只能用两个参数表示我们的模型：`W`和`b`。

目标是生成 pytorch 代码，它能够找到最佳的参数`W`和`b`，它来自输入数据`x_data`，将其拟合到输出数据`y_data`，我们这里它是一条直线，由`y_data = W * x_data + b`定义。读者知道`W`应接近 0.1 且`b`为 0.3，但 TensorFlow 不知道它，必须自己实现。

解决此类问题的一种标准方法是，遍历数据集的每个值并修改参数`W`和`b`，以便每次都能获得更精确的答案。为了确定我们是否在这些迭代中有所改进，我们将定义一个损失函数（也称为“误差函数”）来衡量某条线有多“好”（实际上是有多“坏”）。

该函数接收参数`W`和`b`，并根据线与数据的拟合程度返回一个误差值。在我们的例子中，我们可以使用均方误差 [14] 作为损失函数。利用均方误差，我们得到“误差”的平均值，基于实际值与算法每次迭代估计值之间距离。

稍后，我将详细介绍损失函数及其替代方法，但对于这个介绍性示例，均方误差有助于我们一步一步向前推进。

现在是时候用 pytorch 编写我 解释过的所有内容了。为此，首先我们将使用以下语句创建三个变量：

```python
import torch
from torch.autograd import Variable
W = Variable(torch.randn(1).uniform_(-1,1)) # Variables wrap a Tensor
b =  Variable(torch.zeros([1]))
x_data = np.array(x_data)
y = W * x_data + b
```

现在，我们可以继续前进，只知道方法`Variable`的调用定义了一个变量，驻留在 pytorch 的内部图数据结构中，我在上面已经说过了。 稍后我们将回到方法参数的更多信息，但是现在我认为最好继续前进来推进第一种方法。

现在，通过定义这些变量，我们可以基于每个点与函数`y = W * x + b`计算的点之间的距离，来表示我们之前讨论的损失函数。之后，我们可以计算其平方和的平均值。 在 pytorch 中，此损失函数表示如下：

```py
loss = torch.nn.MSELoss()
y = torch.tensor(y)
y_data = torch.tensor(y_data)

output = loss(y,y_data)
```

如我们所见，此表达式计算我们知道的`y_data`点与从输入`x_data`计算的点`y`之间的平方距离的平均值。

此时，读者可能已经怀疑最适合我们数据的直线是误差值较小的直线。 因此，如果我们使误差函数最小，我们将找到我们数据的最佳模型。

目前没有太多细节，这就是使函数最小的优化算法，称为梯度下降 [15]。 理论上，梯度下降是一种算法，它接受由一组参数定义的函数，它以一组初始参数值开始，并迭代地移向一组使函数最小的值。 在函数梯度 [16] 的负方向上移动来实现迭代式最小化。 通常计算距离平方来确保它是正的并且使误差函数可微分以便计算梯度。

算法从一组参数的初始值开始（在我们的例子中为`W`和`b`），然后算法以某种方式迭代地调整这些变量的值，在过程结束时，变量的值使成本函数最小。

要在 pytorch 中使用此算法，我们只需执行以下两个语句：

```py
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
```

现在，这足以让 TensorFlow 在其内部数据结构中创建相关数据，并且在这个结构中也实现了一个可以由`train`调用的优化器，它是针对定义的成本函数的梯度下降算法。稍后，我们将讨论名为学习率的函数参数（在我们的示例中，值为 0.5）。

### 运行算法

正如我们之前所见，在代码的这个位置上，特定于 TensorFlow 库的调用，只向其内部图添加了信息，而 TensorFlow 的运行时尚未运行任何算法。因此，与前一章的示例一样，我们必须创建会话，调用`run`方法并传递`train`作为参数。另外，因为在代码中我们已经指定了变量，所以我们必须先使用以下调用对它们进行初始化：

```py
init = tf.initialize_all_variables()
 
sess = tf.Session()
sess.run(init)
```

现在我们可以开始迭代过程，这将允许我们找到`W`和`b`的值，它定义最适合输入点的模型直线。 训练过程一直持续到模型在训练数据上达到所需的准确度。 在我们的特定示例中，如果我们假设只有 8 次迭代就足够了，代码可能是：

```py
for step in xrange(8):
   sess.run(train)
print step, sess.run(W), sess.run(b)
```

运行此代码的结果表明，`W`和`b`的值接近我们事先知道的值。 在我的例子中，`print`的结果是：

```py
(array([ 0.09150752], dtype=float32), array([ 0.30007562], dtype=float32))
```

并且，如果我们使用以下代码以图形方式显示结果：

```py
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()
```

我们可以用图形方式，看到参数`W = 0.0854`和`b = 0.299`定义的直线，只需 8 次迭代：

![](https://jorditorres.org/wp-content/uploads/2016/02/image016.png)

请注意，我们只执行了八次迭代来简化说明，但如果我们运行更多，参数值会更接近预期值。 我们可以使用以下语句来打印`W`和`b`的值：

```py
print(step, sess.run(W), sess.run(b))
```

在我们的例子中，`print`输出是：

```py
(0, array([-0.04841119], dtype=float32), array([ 0.29720169], dtype=float32))
(1, array([-0.00449257], dtype=float32), array([ 0.29804006], dtype=float32))
(2, array([ 0.02618564], dtype=float32), array([ 0.29869056], dtype=float32))
(3, array([ 0.04761609], dtype=float32), array([ 0.29914495], dtype=float32))
(4, array([ 0.06258646], dtype=float32), array([ 0.29946238], dtype=float32))
(5, array([ 0.07304412], dtype=float32), array([ 0.29968411], dtype=float32))
(6, array([ 0.08034936], dtype=float32), array([ 0.29983902], dtype=float32))
(7, array([ 0.08545248], dtype=float32), array([ 0.29994723], dtype=float32))
```

你可以观察到算法以`W = -0.0484`和`b = 0.2972`（在我们的例子中）的初始值开始，然后算法以一种方式迭代调整变量的值使损失函数最小。

你还可以检查损失函数是否随之减少

```py
print(step, sess.run(loss))
```

在这种情况下，`print`输出是：

```py
(0, 0.015878126)
(1, 0.0079048825)
(2, 0.0041520335)
(3, 0.0023856456)
(4, 0.0015542418)
(5, 0.001162916)
(6, 0.00097872759)
(7, 0.00089203351)
```

我建议读者在每次迭代时绘图，让我们可以直观地观察算法如何调整参数值。 在我们的例子中，8 个截图是：

![](https://jorditorres.org/wp-content/uploads/2016/02/image018.png)

正如读者可以看到的，在算法的每次迭代中，直线更适合数据。 梯度下降算法如何更接近最小化损失函数的参数值？

由于我们的误差函数由两个参数（`W`和`b`）组成，我们可以将它可视化为二维表面。 该二维空间中的每个点代表一条直线。 每个点的函数高度是该直线的误差值。 在该表面上，一些直线产生的误差值小于其他直线。 当 TensorFlow 运行梯度下降搜索时，它将从该表面上的某个位置开始（在我们的示例中，点`W = -0.04841119`和`b = 0.29720169`）并向下移动来查找具有最小误差的直线。

要在此误差函数上运行梯度下降，TensorFlow 会计算其梯度。 梯度将像指南针一样，总是引导我们向下走。 为了计算它，TensorFlow 将对误差函数微分，在我们的情况下意味着它需要计算`W`和`b`的偏导数，它表明每次迭代中要移动的方向。

之前提到的学习率参数控制每次迭代期间 TensorFlow 的每一步的下降程度。 如果我们引入的参数太大，我们可能会越过最小值。 但是，如果我们让 TensorFlow 采取较小步骤，则需要多次迭代才能达到最小值。 因此，使用良好的学习率至关重要。 有不同的技术来调整学习率参数的值，但它超出了本入门书的范围。 确保梯度下降算法正常工作的一种好方法，是确保每次迭代中的误差减小。

请记住，为了便于读者测试本章所述的代码，你可以从本书的 Github [17] 下载`regression.py`。 在这里，你将发现所有东西都在一起以便跟踪：

```py
import numpy as np

num_points = 1000
vectors_set = []
for i in xrange(num_points):
         x1= np.random.normal(0.0, 0.55)
         y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

import matplotlib.pyplot as plt

#Graphic display
plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.show()

import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(8):
     sess.run(train)
     print(step, sess.run(W), sess.run(b))
     print(step, sess.run(loss))

     #Graphic display
     plt.plot(x_data, y_data, 'ro')
     plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
     plt.xlabel('x')
     plt.xlim(-2,2)
     plt.ylim(0.1,0.6)
     plt.ylabel('y')
     plt.legend()
     plt.show()
```

在本章中，我们已经开始探索 TensorFlow 软件包的可能性，首先采用直观方法处理两个基本组件：损失函数和梯度下降算法，使用基本线性回归算法来介绍。 在下一章中，我们将详细介绍 TensorFlow 包使用的数据结构。
