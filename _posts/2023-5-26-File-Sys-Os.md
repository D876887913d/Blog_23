---
layout: posts
title:  "File os sys function"
date:   2023-05-26 8:24:22 +0800
categories: jekyll update
---

```python
import os
import sys

# getcwd是获取当前的文件目录
# split是将文件名与前面的文件目录分开，然后一起放到一个元组里
# 第[0]项为前面的路径，第[1]项为后面的文件名
py_dir = os.path.split(os.getcwd())[0]

if py_dir not in sys.path:
    sys.path.append(py_dir)
```


```python
os.getcwd()
```




    'd:\\project_code\\python_project\\Jetseg'




```python
os.path.split(os.getcwd())
```




    ('d:\\project_code\\python_project', 'Jetseg')




```python
# 这个好像是系统环境变量的位置
sys.path
```




    ['d:\\project_code\\python_project\\Jetseg',
     'd:\\Anaconda\\envs\\pytorch\\python37.zip',
     'd:\\Anaconda\\envs\\pytorch\\DLLs',
     'd:\\Anaconda\\envs\\pytorch\\lib',
     'd:\\Anaconda\\envs\\pytorch',
     '',
     'C:\\Users\\gwj\\AppData\\Roaming\\Python\\Python37\\site-packages',
     'd:\\Anaconda\\envs\\pytorch\\lib\\site-packages',
     'd:\\Anaconda\\envs\\pytorch\\lib\\site-packages\\win32',
     'd:\\Anaconda\\envs\\pytorch\\lib\\site-packages\\win32\\lib',
     'd:\\Anaconda\\envs\\pytorch\\lib\\site-packages\\Pythonwin',
     'd:\\Anaconda\\envs\\pytorch\\lib\\site-packages\\IPython\\extensions',
     'C:\\Users\\gwj\\.ipython',
     'd:\\project_code\\python_project']




```python
# 返回的是执行环境的具体内容
print('Arguments:', sys.argv)
```

    Arguments: ['d:\\Anaconda\\envs\\pytorch\\lib\\site-packages\\ipykernel_launcher.py', '--ip=127.0.0.1', '--stdin=9003', '--control=9001', '--hb=9000', '--Session.signature_scheme="hmac-sha256"', '--Session.key=b"3c83c301-9eaf-43c5-a64b-689a2949eff5"', '--shell=9002', '--transport="tcp"', '--iopub=9004', '--f=c:\\Users\\gwj\\AppData\\Roaming\\jupyter\\runtime\\kernel-v2-16468dOSdI0pIn3Si.json']
    


```python
# 获取文件的大小
sys.getsizeof(".gitignore")
```




    59




```python
# 最大回归深度，对于无限递归的函数，最终虽然会把栈用完掉，但是要尽可能避免用完栈再停止。
print('Initial limit:', sys.getrecursionlimit())
# 修改最大回归深度
# sys.setrecursionlimit(10)
print('Modified limit:', sys.getrecursionlimit())
print()

# maxsize 是列表、字典、字符串或其他数据结构的最大大小，由 C 解释器的大小类型决定。
# maxunicode 是解释器当前配置的最大整数 Unicode 点。
print('maxsize   :', sys.maxsize)
print('maxunicode:', sys.maxunicode)
print()

# 最小的正浮点数
print('Smallest difference (epsilon):', sys.float_info.epsilon)
print()

# 小数点数：15
# 最多位数：53
print('Digits (dig)              :', sys.float_info.dig)
print('Mantissa digits (mant_dig):', sys.float_info.mant_dig)
print()
# 最大、最小的浮点数
print('Maximum (max):', sys.float_info.max)
print('Minimum (min):', sys.float_info.min)
print()

# 剩下的也基本是浮点数的一些性质
print('Radix of exponents (radix):', sys.float_info.radix)
print()
print('Maximum exponent for radix (max_exp):',
      sys.float_info.max_exp)
print('Minimum exponent for radix (min_exp):',
      sys.float_info.min_exp)
print()
print('Max. exponent power of 10 (max_10_exp):',
      sys.float_info.max_10_exp)
print('Min. exponent power of 10 (min_10_exp):',
      sys.float_info.min_10_exp)
print()
print('Rounding for addition (rounds):', sys.float_info.rounds)
```

    Initial limit: 3000
    Modified limit: 3000
    maxsize   : 9223372036854775807
    maxunicode: 1114111
    Smallest difference (epsilon): 2.220446049250313e-16
    
    Digits (dig)              : 15
    Mantissa digits (mant_dig): 53
    
    Maximum (max): 1.7976931348623157e+308
    Minimum (min): 2.2250738585072014e-308
    
    Radix of exponents (radix): 2
    
    Maximum exponent for radix (max_exp): 1024
    Minimum exponent for radix (min_exp): -1021
    
    Max. exponent power of 10 (max_10_exp): 308
    Min. exponent power of 10 (min_10_exp): -307
    
    Rounding for addition (rounds): 1
    


```python

```
