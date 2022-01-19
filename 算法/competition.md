# 题目：5186. 区间内查询数字的频率 显示英文描述 
 
请你设计一个数据结构，它能求出给定子数组内一个给定值的 频率 。

子数组中一个值的 频率 指的是这个子数组中这个值的出现次数。

请你实现 RangeFreqQuery 类：

RangeFreqQuery(int[] arr) 用下标从 0 开始的整数数组 arr 构造一个类的实例。
int query(int left, int right, int value) 返回子数组 arr[left...right] 中 value 的 频率 。
一个 子数组 指的是数组中一段连续的元素。arr[left...right] 指的是 nums 中包含下标 left 和 right 在内 的中间一段连续元素。

 

示例 1：

输入：
["RangeFreqQuery", "query", "query"]
[[[12, 33, 4, 56, 22, 2, 34, 33, 22, 12, 34, 56]], [1, 2, 4], [0, 11, 33]]
输出：
[null, 1, 2]

解释：
RangeFreqQuery rangeFreqQuery = new RangeFreqQuery([12, 33, 4, 56, 22, 2, 34, 33, 22, 12, 34, 56]);
rangeFreqQuery.query(1, 2, 4); // 返回 1 。4 在子数组 [33, 4] 中出现 1 次。
rangeFreqQuery.query(0, 11, 33); // 返回 2 。33 在整个子数组中出现 2 次。

## 答案

```python
class RangeFreqQuery(object):

    def __init__(self, arr):
        """
        :type arr: List[int]
        """
        vs = {}
        for i, v in enumerate(arr):
            if v not in vs: vs[v]=[]
            vs[v].append(i)
        self.ix = vs


    def query(self, left, right, value):
        """
        :type left: int
        :type right: int
        :type value: int
        :rtype: int
        """
        if value not in self.ix: return 0
        a = self.ix[value]
        return bisect_right(a, right)-bisect_left(a, left)



# Your RangeFreqQuery object will be instantiated and called as such:
# obj = RangeFreqQuery(arr)
# param_1 = obj.query(left,right,value)
```
# 题目： 5941. 找出知晓秘密的所有专家
给你一个整数 n ，表示有 n 个专家从 0 到 n - 1 编号。另外给你一个下标从 0 开始的二维整数数组 meetings ，其中 meetings[i] = [xi, yi, timei] 表示专家 xi 和专家 yi 在时间 timei 要开一场会。一个专家可以同时参加 多场会议 。最后，给你一个整数 firstPerson 。

专家 0 有一个 秘密 ，最初，他在时间 0 将这个秘密分享给了专家 firstPerson 。接着，这个秘密会在每次有知晓这个秘密的专家参加会议时进行传播。更正式的表达是，每次会议，如果专家 xi 在时间 timei 时知晓这个秘密，那么他将会与专家 yi 分享这个秘密，反之亦然。

秘密共享是 瞬时发生 的。也就是说，在同一时间，一个专家不光可以接收到秘密，还能在其他会议上与其他专家分享。在所有会议都结束之后，返回所有知晓这个秘密的专家列表。你可以按 任何顺序 返回答案。

 

示例 1：
```
输入：n = 6, meetings = [[1,2,5],[2,3,8],[1,5,10]], firstPerson = 1
输出：[0,1,2,3,5]
解释：
时间 0 ，专家 0 将秘密与专家 1 共享。
时间 5 ，专家 1 将秘密与专家 2 共享。
时间 8 ，专家 2 将秘密与专家 3 共享。
时间 10 ，专家 1 将秘密与专家 5 共享。
因此，在所有会议结束后，专家 0、1、2、3 和 5 都将知晓这个秘密。
```
示例 2：
```
输入：n = 4, meetings = [[3,1,3],[1,2,2],[0,3,3]], firstPerson = 3
输出：[0,1,3]
解释：
时间 0 ，专家 0 将秘密与专家 3 共享。
时间 2 ，专家 1 与专家 2 都不知晓这个秘密。
时间 3 ，专家 3 将秘密与专家 0 和专家 1 共享。
因此，在所有会议结束后，专家 0、1 和 3 都将知晓这个秘密。
```
示例 3：
```
输入：n = 5, meetings = [[3,4,2],[1,2,1],[2,3,1]], firstPerson = 1
输出：[0,1,2,3,4]
解释：
时间 0 ，专家 0 将秘密与专家 1 共享。
时间 1 ，专家 1 将秘密与专家 2 共享，专家 2 将秘密与专家 3 共享。
注意，专家 2 可以在收到秘密的同一时间分享此秘密。
时间 2 ，专家 3 将秘密与专家 4 共享。
因此，在所有会议结束后，专家 0、1、2、3 和 4 都将知晓这个秘密。
```
示例 4：
```
输入：n = 6, meetings = [[0,2,1],[1,3,1],[4,5,1]], firstPerson = 1
输出：[0,1,2,3]
解释：
时间 0 ，专家 0 将秘密与专家 1 共享。
时间 1 ，专家 0 将秘密与专家 2 共享，专家 1 将秘密与专家 3 共享。
因此，在所有会议结束后，专家 0、1、2 和 3 都将知晓这个秘密。
```

提示：
```
2 <= n <= 105
1 <= meetings.length <= 105
meetings[i].length == 3
0 <= xi, yi <= n - 1
xi != yi
1 <= timei <= 105
1 <= firstPerson <= n - 1
```
## 答案
```python
class UnionFind:
    def __init__(self, n):
        self.count = n
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            if self.rank[rootx] < self.rank[rooty]:
                rootx, rooty = rooty, rootx
            self.parent[rooty] = rootx
            if self.rank[rootx] == self.rank[rooty]:
                self.rank[rootx] += 1
            self.count -= 1

    def isolate(self, x):
        self.parent[x] = x
        self.rank[x] = 0


class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:

        meetings.sort(key=lambda x: x[2])

        uf = UnionFind(n)
        uf.union(0, firstPerson)

        for _, members in groupby(meetings, key=lambda x: x[2]):
            members = list(members)

            # 连接同一时间开会的专家
            people = set()
            for x, y, _ in members:
                uf.union(x, y)
                people.add(x)
                people.add(y)

            # 开完会后, 孤立所有没知道秘密的专家
            for person in people:
                if uf.find(person) != uf.find(0):
                    uf.isolate(person)

        ans = []
        for i in range(n):
            if uf.find(i) == uf.find(0):
                ans.append(i)
        return ans

```
# 5955. 摘水果 显示英文描述 
 
在一个无限的 x 坐标轴上，有许多水果分布在其中某些位置。给你一个二维整数数组 fruits ，其中 fruits[i] = [positioni, amounti] 表示共有 amounti 个水果放置在 positioni 上。fruits 已经按 positioni 升序排列 ，每个 positioni 互不相同 。

另给你两个整数 startPos 和 k 。最初，你位于 startPos 。从任何位置，你可以选择 向左或者向右 走。在 x 轴上每移动 一个单位 ，就记作 一步 。你总共可以走 最多 k 步。你每达到一个位置，都会摘掉全部的水果，水果也将从该位置消失（不会再生）。

返回你可以摘到水果的 最大总数 。

 

示例 1：

```
输入：fruits = [[2,8],[6,3],[8,6]], startPos = 5, k = 4
输出：9
解释：
最佳路线为：
- 向右移动到位置 6 ，摘到 3 个水果
- 向右移动到位置 8 ，摘到 6 个水果
移动 3 步，共摘到 3 + 6 = 9 个水果
```
![image](https://user-images.githubusercontent.com/69283174/145699853-e1b6f574-3edd-4a3b-9cff-4d10421122fe.png)



示例 2：

```
输入：fruits = [[0,9],[4,1],[5,7],[6,2],[7,4],[10,9]], startPos = 5, k = 4
输出：14
解释：
可以移动最多 k = 4 步，所以无法到达位置 0 和位置 10 。
最佳路线为：
- 在初始位置 5 ，摘到 7 个水果
- 向左移动到位置 4 ，摘到 1 个水果
- 向右移动到位置 6 ，摘到 2 个水果
- 向右移动到位置 7 ，摘到 4 个水果
移动 1 + 3 = 4 步，共摘到 7 + 1 + 2 + 4 = 14 个水果
```
![image](https://user-images.githubusercontent.com/69283174/145699856-4d2b611b-04b7-4cca-883e-584bcb974fed.png)


示例 3：

```
输入：fruits = [[0,3],[6,4],[8,5]], startPos = 3, k = 2
输出：0
解释：
最多可以移动 k = 2 步，无法到达任一有水果的地方
```
![image](https://user-images.githubusercontent.com/69283174/145699858-422d44df-bd42-4436-bfc9-63a4cd440fe6.png)

在满足总步数小于 k 的条件下，只需要确定最大值的左右端点即可。
首先先从左边开始，找到能够到达的最远点，并把从最远点开始到 startPos 的所有水果加起来，同时入队。
然后我们就要开始确定右端点，在确定右端点的同时，可能会改变左端点的位置。这就要求我们不断检查左右端点之间的步数，记左右端点分别为a ,b，那么步数stepstep为b - a + min(startPos - a, b - startPos)，可以自己推一推。如果step <= k，那么直接加上右端点的值，否则就要减去左端的值，并同时pop出队列，然后再次对左端点做上述检查。
遍历的中止条件是，直到右端点b - startPos > k，也就是到达不了b即可停止，
 
提示：
```
1 <= fruits.length <= 105
fruits[i].length == 2
0 <= startPos, positioni <= 2 * 105
对于任意 i > 0 ，positioni-1 < positioni 均成立（下标从 0 开始计数）
1 <= amounti <= 104
0 <= k <= 2 * 105
```
```python
class Solution:
    def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
        h = deque([])
        res = 0
        n = len(fruits)
        p = 0
        #先从左边开始，找到能够到达的最远点，并把从最远的开始到startPos的所有水果加起来，同时入队
        while p < n and fruits[p][0] <= startPos:
            if abs(fruits[p][0] - startPos) <= k:
                res += fruits[p][1]
                h.append((fruits[p][0], fruits[p][1]))
            p += 1

        tmp = res
        while p < n and fruits[p][0] - startPos <= k:
            #对于每一个startPos右端的水果，依次检查左端点是否满足条件
            while h and h[0][0] < startPos and fruits[p][0] - h[0][0] + min(startPos - h[0][0], fruits[p][0] - startPos) > k:
                tmp -= h[0][1]
                h.popleft()
            tmp += fruits[p][1]
            res = max(res, tmp)
            p += 1

        return res 
```
