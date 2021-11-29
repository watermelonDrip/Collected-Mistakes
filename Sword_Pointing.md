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

 
 
