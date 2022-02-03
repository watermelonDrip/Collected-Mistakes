# 堆的定义和分类

堆 是一种特别的二叉树，满足以下条件的二叉树，可以称之为堆：
+ 完全二叉树；
+ 每一个节点的值都必须 大于等于或者小于等于 其孩子节点的值。

## 堆的实现
我们可以用数组来实现堆。我们将堆中的元素以二叉树的形式存在数组中。
```python
# 「最小堆」的实现
import sys

class MinHeap:
    def __init__(self, heapSize):
        # heapSize用于数组的大小，因为数组在创建的时候至少需要指明数组的元素个数
        self.heapSize = heapSize
        # 使用数组创建完全二叉树的结构，然后使用二叉树构建一个「堆」
        self.minheap = [0]*(heapSize+1)
        # realSize用于记录「堆」的元素个数
        self.realSize = 0

    #  添加元素函数
    def add(self, element):
        self.realSize += 1
        # 如果「堆」中元素的个数大于一开始设定的数组的个数，则返回「Add too many elements」
        if self.realSize > self.heapSize:
            print("Add too many elements!")
            self.realSize -= 1
            return
        # 将添加的元素添加到数组中
        self.minheap[self.realSize] = element
        # 新增元素的索引位置
        index = self.realSize
        # 新增元素的父节点的索引位置
        # 注意，如果用数组表示完全二叉树，并且根结点存储在数组的索引1的位置的时候，任何一个节点的父节点索引位置为「该节点的索引位置/2」，任何一个节点的左孩子节点的索引位置为「该节点的索引位置*2」，任何一个节点的右孩子节点的索引位置为「该节点的索引位置*2+1」
        parent = index // 2
        # 当添加的元素小于父节点时，需要将父节点的值和新增元素的值交换
        while (self.minheap[index] < self.minheap[parent] and index > 1):
            self.minheap[parent], self.minheap[index] = self.minheap[index], self.minheap[parent]
            index = parent
            parent = index // 2
    
    # 获取堆顶元素函数
    def peek(self):
        return self.minheap[1]
    
    # 删除堆顶元素函数
    def pop(self):
        # 如果当前「堆」的元素个数为0， 则返回「Don't have any element」
        if self.realSize < 1:
            print("Don't have any element!")
            return sys.maxsize
        else:
            # 当前「堆」中含有元素
            # self.realSize >= 1
            removeElement = self.minheap[1]
            # 将「堆」中的最后一个元素赋值给堆顶元素
            self.minheap[1] = self.minheap[self.realSize]
            self.realSize -= 1
            index = 1
            # 当删除的元素不是孩子节点时
            while (index < self.realSize and index <= self.realSize // 2):
                # 被删除节点的左孩子节点
                left = index * 2
                # 被删除节点的右孩子节点
                right = (index * 2) + 1
                # 当删除节点的元素大于 左孩子节点或者右孩子节点，代表该元素的值大，此时需要将该元素与左、右孩子节点中最小的值进行交换
                if (self.minheap[index] > self.minheap[left] or self.minheap[index] > self.minheap[right]):
                    if self.minheap[left] < self.minheap[right]:
                        self.minheap[left], self.minheap[index] = self.minheap[index], self.minheap[left]
                        index = left
                    else:
                        self.minheap[right], self.minheap[index] = self.minheap[index], self.minheap[right]
                        index = right
                else:
                    break
            return removeElement
    
    # 返回「堆」的元素个数
    def size(self):
        return self.realSize
    
    def toString(self):
        print(self.minheap[1 : self.realSize+1])
        

if __name__ == "__main__":
    	# 测试用例
        minHeap = MinHeap(5)
        minHeap.add(3)
        minHeap.add(1)
        minHeap.add(2)
        # [1,3,2]
        minHeap.toString()
        # 1
        print(minHeap.peek())
        # 1
        print(minHeap.pop())
        # 2
        print(minHeap.pop())
        # 3
        print(minHeap.pop())
        minHeap.add(4)
        minHeap.add(5)
        # [4,5]
        minHeap.toString()

```
## 堆排序
堆排序指的是利用堆的数据结构对一组无序元素进行排序。

### Top K 问题 - 解法 1

 
### Top K 问题 - 解法 2

 
