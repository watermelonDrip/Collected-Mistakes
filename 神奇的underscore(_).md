1.  存储上个value
```python
>>> 10 
10 
>>> _ 
10 
>>> _ * 3 
30 
>>> _ * 20 
600
```

2. 忽略一些变量
```python
# Ignore a value when unpacking
x, _, y = (1, 2, 3) # x = 1, y = 3 
```

3. 给变量和function 起名
+ single_leading_underscore: Python并没有真正的私有化支持，但可用下划线得到伪私有。以单下划线开头的表示的是protected类型的变量。 
即保护类型只能允许其本身与子类进行访问。若内部变量标示，如： 当使用“from M import”时，不会将以一个下划线开头的对象引入 。
```python
_internal_name = 'one_nodule' # private variable
_internal_version = '1.0' # private variable

class _Base: # private class
    _hidden_factor = 2 # private variable
    def __init__(self, price):
        self._price = price
    def _double_price(self): # private method
        return self._price * self._hidden_factor    
    def get_double_price(self):
        return self._double_price() 
```
+ single_trailing_underscore: 避免和Python 的build-ins 和keywords 产生冲突。（其实很少用）

```python
Tkinter.Toplevel(master, class_='ClassName') # Avoid conflict with 'class' keyword
```
```python
list_ = List.objects.get(1) # Avoid conflict with 'list' built-in type
```

+ double_leading_underscore: 

4. Python中，实例的变量名如果以__开头（双下划线），就变成了一个私有变量（private），只有内部可以访问，外部不能访问
5. 特殊意义的变量
6. 隔离字符
