---
layout: posts
title:  "Debug Jetseg"
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
DEBUG的过程中，由于觉得nb文件不太好DEBUG，就去调用了py文件，调用Py文件的过程中，报错no module named ...的情况，采用了：`sys.path.append(r"D:\project_code\python_project\Jetseg")`添加了工作目录后，不再报错，可以正常DEBUG。具体代码位置如下：
```
py/train.py


import pathmagic

import os
import gc
import sys
import logging
import argparse

import torch
import torch.cuda as cuda

>> sys.path.append(r"D:\project_code\python_project\Jetseg")

from modules.model import JetSeg
from modules.utils import _init_logger, get_path, pkl, ModelConfigurator

from modules.mltools import (
    load_dataset, load_dataloaders, build_loss_fn,
    build_optimizer, training
)

```
然后debug过程中遇到了这个BUG：
```cmd
Exception has occurred: FileNotFoundError
[Errno 2] No such file or directory: 'D:\\project_code\\python_project/train/JetSeg-M3-C0-camvid-global-losses.csv'
```
我觉得不存在文件，那么进行新建文件，即可解决问题。参考的判断python文件是否存在的代码来源是:[solution](https://www.freecodecamp.org/news/how-to-check-if-a-file-exists-in-python/).新建文件采用的代码是：
```python
>>> import pathlib
>>> pathlib.Path("./5.txt").touch()
```
修改之后不再报错。然后在进行DEBUG过程中，发现encoder出来的结果是一个[]，因此需要进行encoder部分的DEBUG。  
```
 UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.
  "Argument interpolation should be of type InterpolationMode instead of int. "
```
找到的解决方案为：  
```python
from torchvision.transforms import InterpolationMode
transforms.Resize(size=(opt.h, opt.w),interpolation=3), #Image.BICUBIC
                                   ||进行相关的修改
                                   ||将参数interpolation函数插值法
transforms.Resize((opt.h, opt.w), interpolation=InterpolationMode.BICUBIC),
```
实际上这个报不报错倒也无所谓，然后有一个会报错csv文件目录不存在，需要自己修改文件路径：
```python
py/train.py

    # Loading dataloaders
    dataloaders = load_dataloaders(dataset, batch_size)
    logger.info('Dataset read')

    # Getting train path
>> line 62  
    train_path = "./train/"

    # Building loss fn and optimizer
    loss_fn = build_loss_fn(loss_name="jet",
                            pixels_per_class=cfg.pixels_per_class)

```
由于中途修改了load_dataset输入的尺寸(修改原因是自己电脑显存太小了),导致后面encoder层进行特征图提取的时候出现了问题，所以需要修改encoder函数，这里先给出具体的load_dataset需要改的地方：
```python
modules/mltools.py
def load_dataset(dataset_name, num_classes):

    # Getting data workspace
    # data_path = get_path('data')
>>  data_path=r"E:\public_data\JetSegUsage\CamVid\\"

    # Getting dataset path
    # data_path += dataset_name.lower() + "/"

    if dataset_name.lower() == 'cityscapes':
        pass

    elif dataset_name.lower() == 'camvid':

        # Getting color map codification
        color_dict = data_path + 'class_dict.csv'
        code2id, id2code, name2id, id2name = color_map(color_dict)
        color_maps = {
            "code2id": code2id,
            "id2code": id2code,
            "name2id": name2id,
            "id2name": id2name,
        }

        train = SSegmDataset(dataset_name=dataset_name.lower(),
                             num_classes=num_classes,
                             root_path=data_path, mode="train",
                             color_map=color_maps,
>>                           img_size=(224,224)
                             )

        test = SSegmDataset(dataset_name=dataset_name.lower(),
                            root_path=data_path, mode="test",
                            num_classes=num_classes,
                            color_map=color_maps,
>>                          img_size=(224,224)
                            )

        valid = SSegmDataset(dataset_name=dataset_name.lower(),
                             root_path=data_path, mode="val",
                             num_classes=num_classes,
                             color_map=color_maps,
>>                           img_size=(224,224)
                             )

        dataset = {}
        dataset["train"] = train
        dataset["test"] = test
        dataset["valid"] = valid

        return dataset

    else:
        raise ValueError(f"The dataset {dataset_name} doesn't exists")

    return dataset

```
进行了尝试1：
```python
def train(train_cfg, logger):

    # Getting train configuration
    dataset_name = train_cfg["dataset_name"]
    jetseg_comb = train_cfg["jsc"]
    jetseg_mode = train_cfg["jsm"]
    num_epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]

    # Create model configuration
    cfg = ModelConfigurator(comb=jetseg_comb, 
    mode=jetseg_mode,
    >> img_sizes=(224,224),
    dataset_name=dataset_name)

    # Build model
    model = JetSeg(cfg)
    model = model.to("cuda")
    logger.info('Model built')
```
这里我其实改完之后有新的报错了，我目前的话有点觉得自己应该从简单的segmentation项目看一下，想找一下基本的分割算法。这部分先暂时缓冲。——by. 2023.5.27 22:54  
其实本来吧，已经放弃了，不过觉得自己还可以再稍微努力一下，这个也并非自己没法解决的。  
DEBUG走起！BCELOSS报的错，最后在输出部分加了一个sigmoid层，不报错了，具体原因的话，应该是BCELoss只支持[0,1]之间的输入，然而之前的jetseg输出包含这个范围外的数值，因此会报错具体修改如下：
```
class JetSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = JetNet(cfg)
        self.decoder = RegSeg(cfg)

        # 如果使用的是BCEloss需要添加一个转换。
>>      self.Binaryly=nn.Sigmoid()

    def forward(self, x):

        # Get encoded feature maps
        encode_fts = self.encoder(x)

        # Get decoded feature maps
        decode_fts = self.decoder(encode_fts)

        # Interpolate decoded output as original img size
        x = F.interpolate(decode_fts, size=x.shape[-2:],
                          mode='bilinear', align_corners=False)
>>      x=self.Binaryly(x)
        return x
```
然后就是疯狂改test.ipynb的过程了，基本就是cfg文件修改+各种小点修改。修改完之后，进行了一次运行。  
最后进行推理采用的是evaluation函数，展示预测结果的函数是用的：
```
show_sample(batch_out[0][0].swapaxes(0,2).swapaxes(1,2))
show_sample(batch_out[0][1].swapaxes(0,2).swapaxes(1,2))
show_sample(batch_out[0][3].swapaxes(0,2).swapaxes(1,2))
```
推理结果暂时就不截图上来了，不好截图到md。

# 中途的一些感受
代码修改的过程中，自己去收集数据集的能力变强了，也有一些自己的小算盘：  
- 收集比较好的代码，供自己之后使用。  
- 把每个领域的数据集收集起来，然后用于之后设计demo。