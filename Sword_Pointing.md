# 数组字符串

## 剑指 Offer 04. 二维数组中的查找
一个二维数组中，每一行和每一列都是递增的顺序排序。 判断数组中是否含有target整数。

### 思路
从矩阵的左下角看，上方的数字都比其小，右方的数字都比其大，所以依据该规律去判断数字是否存在
###
```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        
        n = len(matrix)
        if n == 0:
            return False
        m = len(matrix[0])
        
        t_n = n-1
        t_m = 0
        while t_n>=0 and t_m<m:
            if matrix[t_n][t_m] == target:
                return True
            elif matrix[t_n][t_m] > target:
                t_n-=1
            else:
                t_m+=1
        return False
```
O(M+N)

## 剑指 Offer 11. 旋转数组的最小数字
输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  
 
## 思路
标签：二分查找
整体思路：首先数组是一个有序数组的旋转，从这个条件可以看出，数组是有大小规律的，可以使用二分查找利用存在的规律快速找出结果
时间复杂度：O(logn) ，空间复杂度：O(1) 
## 算法流程
+ 初始化下标 left 和 right
+ 计算中间下标 mid = (right + left) / 2​，这里的除法是取整运算，不能出现小数
+ 当 numbers[mid] < numbers[right] 时，说明最小值在 ​[left, mid]​ 区间中，则令 right = mid，用于下一轮计算
+ 当 numbers[mid] > numbers[right]​ 时，说明最小值在 [mid, right]​ 区间中，则令 left = mid + 1，用于下一轮计算
+ 当 numbers[mid] == numbers[right]​ 时，无法判断最小值在哪个区间之中，此时让 right--，缩小区间范围，在下一轮进行判断
+ 为什么是 right-- 缩小范围，而不是 left++？
    + 因为数组是升序的，所以最小值一定靠近左侧，而不是右侧。比如，当存在 [1,2,2,2,2] 这种情况时，left = 0，right = 4，mid = 2，数值满足 numbers[mid] == numbers[right] 这个条件，如果 left++，则找不到最小值

##
```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left = 0
        right = len(numbers)-1
        while left<= right:  #左闭右闭
            mid = (left + right)//2
            if numbers[mid] > numbers[right]: # 最小值肯定不是numbers[mid], 因此范围改为在[mid+1,right]中存在最小值
                left = mid +1
            elif numbers[mid] < numbers[right]: # 最小值可能是numbers[mid],因此范围改为[left,mid]中存在最小值
                right = mid
            else:  # 实际上，上两个条件，我们是要在范围内， 保持数组是升序排序，所以，我们需要的最小数一定是靠左的，因此当numbers[mid] == numbers[right]，right-=1
                right-=1
        return numbers[left]
```
## 剑指 Offer 29. 顺时针打印矩阵
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

### 思路
```
               top
            1 1 1 1 1
      left  1 1 1 1 1  right
            1 1 1 1 1
            1 1 1 1 1
             bottom
```

### 

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        res = list()
        left,right = 0, len(matrix[0])-1
        top,bottom = 0, len(matrix)-1
        x = 0
        while True:
            for i in range(left,right+1): res.append(matrix[top][i])
            top +=1
            if top> bottom: break

            for i in range(top,bottom+1): res.append(matrix[i][right])
            right-=1
            if left>right:break

            for i in range(right,left-1,-1): res.append(matrix[bottom][i])
            bottom-=1
            if bottom<top: break

            for i in range(bottom,top-1,-1): res.append(matrix[i][left])
            left+=1
            if left>right: break
        return res
```
 
## 剑指 Offer 39. 数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

### 思路


## 剑指 Offer 45. 把数组排成最小的数

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

<!-- https://excalidraw.com/#json=ejqYS52UNAEdw-JoPJvLZ,BYKGVFB67BLkDXBE6jU8Mg -->
![image](https://user-images.githubusercontent.com/69283174/144213553-a751cc12-a90d-4709-8dc0-6e54fbbe5e8a.png)


### 

