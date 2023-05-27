---
layout: posts
title:  "Debug FCOS"
date:   2023-05-26 18:12:22 +0800
categories: jekyll update
---

# 开始前的一些感受
自己真的是想做的事情很多，也知道什么事情是最重要的，说到底就是没法动手去做，只知道这件事情能够让自己体验变好，但是就不去做，真的是，难以克服这件事情的话，肯定也很难变得像众多优秀的开发者一样优秀吧。

# 搭建实例 
给定的autodl没有一些Python库，所以需要安装一下：
```
pip install pandas
pip install natsort
pip install seaborn
pip install torchinfo 
```
顺着运行的时候报错：
`TypeError: __init__() got an unexpected keyword argument 'last_fmap'`
把这个参数删除，不再报错。  
顺着运行又遇到了：`RuntimeError: could not create a descriptor for a dilated convolution forward propagation primitive`  
然后就是缺少数据集了，直接去Kaggle搜索现成的，虽然也可以处理数据集，不过这部分之后再学也不迟。  
根据代码所写，其中的label_classes.csv文件里面列名classes替换为了name.秉持着下一次重开更方便的态度，我将从官网上复制下来的csv文件的列名改的跟它的一致。  
在进行的时候又发现了一个kaggle上的数据集，更符合它的要求，所以就下载了，下载之后发现代码还是没法运行，有一个pandas部分的需求：
```
    cls = pd.read_csv(csv_file)
    color_code = [tuple(cls.drop("name", axis=1).loc[idx]) for idx in range(cls.shape[1])]
    code2id = {v: k for k, v in enumerate(list(color_code))}
    id2code = {k: v for k, v in enumerate(list(color_code))}

    color_name = [cls['name'][idx] for idx in range(cls.shape[1])]
    name2id = {v: k for k, v in enumerate(list(color_name))}
    id2name = {k: v for k, v in enumerate(list(color_name))}
```
报错：`KeyError: "['name'] not found in axis"`  
从这个代码看，我觉得他就是想弄出来后面几个表示颜色的数据，切片就可以实现。采用iloc函数即可：  
```
df.iloc[行切片，列切片]
```
summary的具体结果在换成设备CUDA后可以正常输出了：  
```
==============================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
==============================================================================================================
JetSeg                                                       [1, 32, 960, 720]         --
├─JetNet: 1-1                                                [1, 64, 15, 12]           --
│    └─ModuleList: 2-1                                       --                        --
│    │    └─JetBlock: 3-1                                    [1, 8, 480, 360]          142
│    │    └─JetBlock: 3-2                                    [1, 16, 240, 180]         834
│    │    └─JetBlock: 3-3                                    [1, 16, 240, 180]         448
│    │    └─JetBlock: 3-4                                    [1, 32, 120, 90]          1,024
│    │    └─JetBlock: 3-5                                    [1, 32, 120, 90]          1,088
│    │    └─JetBlock: 3-6                                    [1, 48, 60, 45]           2,304
│    │    └─JetBlock: 3-7                                    [1, 48, 60, 45]           1,107
│    │    └─JetBlock: 3-8                                    [1, 64, 30, 23]           1,091
│    │    └─JetBlock: 3-9                                    [1, 64, 30, 23]           1,475
│    │    └─JetBlock: 3-10                                   [1, 64, 15, 12]           963
│    │    └─JetBlock: 3-11                                   [1, 64, 15, 12]           1,475
│    │    └─JetBlock: 3-12                                   [1, 64, 15, 12]           963
├─RegSeg: 1-2                                                [1, 32, 480, 360]         --
│    └─ModuleList: 2-2                                       --                        --
│    │    └─DecoderHead: 3-13                                [1, 64, 15, 12]           4,224
│    │    └─DecoderHead: 3-14                                [1, 64, 120, 90]          2,176
│    │    └─DecoderHead: 3-15                                [1, 64, 480, 360]         640
│    └─CBA: 2-3                                              [1, 64, 480, 360]         --
│    │    └─Conv2d: 3-16                                     [1, 64, 480, 360]         4,096
│    │    └─BatchNorm2d: 3-17                                [1, 64, 480, 360]         128
│    │    └─REU: 3-18                                        [1, 64, 480, 360]         --
│    └─CBA: 2-4                                              [1, 64, 480, 360]         --
│    │    └─Conv2d: 3-19                                     [1, 64, 480, 360]         8,192
│    │    └─BatchNorm2d: 3-20                                [1, 64, 480, 360]         128
│    │    └─REU: 3-21                                        [1, 64, 480, 360]         --
│    └─Sequential: 2-5                                       [1, 32, 480, 360]         --
│    │    └─Conv2d: 3-22                                     [1, 32, 480, 360]         2,080
==============================================================================================================
Total params: 34,578
Trainable params: 34,578
Non-trainable params: 0
Total mult-adds (G): 2.75
==============================================================================================================
Input size (MB): 8.29
Forward/backward pass size (MB): 861.00
Params size (MB): 0.14
Estimated Total Size (MB): 869.44
==============================================================================================================
```
由于云服务器报错异步什么，所以从自己电脑上试着进行DEBUG：
```
RuntimeError: CUDA out of memory. Tried to allocate 338.00 MiB (GPU 0; 2.00 GiB total capacity; 1.44 GiB already allocated; 0 bytes free; 1.50 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
显存爆了。  
要么加transforms，要么又得回到云服务器。为了尽可能先理解他的思路，于是选择回到云服务器。不过考虑到云服务器的显存也爆了的情况，于是决定首先先编写一下处理图片尺寸的代码。
# 中途的一些感受
代码修改的过程中，自己去收集数据集的能力变强了，也有一些自己的小算盘：  
- 收集比较好的代码，供自己之后使用。  
- 把每个领域的数据集收集起来，然后用于之后设计demo。