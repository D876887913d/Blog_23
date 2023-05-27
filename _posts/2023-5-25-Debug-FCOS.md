---
layout: posts
title:  "Debug FCOS"
date:   2023-05-25 23:46:22 +0800
categories: jekyll update
---

# Linux Command
- pwd 获取当前位置的绝对路径


# Python Coding
如果报错找不到某个Module
```
import sys
sys.path.append("the needed dependent file's path")
```
记录一下INSTALL各个操作的反馈吧：
```
<!-- 创建新的环境，然后激活这个环境 -->
conda create --name FCOS
conda activate FCOS
<!-- 正常安装几个Python库 -->
conda install ipython
pip install ninja yacs cython matplotlib tqdm
<!-- 安装大头,torch+CUDNN -->
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
<!-- 这上面的话，我没autodl的管理员身份，感觉会引起环境冲突，就没乱下，应该也都能用，下不下无所谓 -->

<!-- 配置环境变量，并将路径移动到该文件夹 -->
export INSTALL_DIR=$PWD
cd $INSTALL_DIR

<!-- 克隆coco数据集的API -->
git clone https://github.com/cocodataset/cocoapi.git
<!-- 移动到对应文件夹里 -->
cd cocoapi/PythonAPI
<!-- 安装 -->
python setup.py build_ext install

<!-- 同上 -->
cd $INSTALL_DIR
git clone https://github.com/tianzhi0549/FCOS.git
cd FCOS
setup
然后就报错了
整不出来，洗洗睡了，明天试试其他几个吧，yolov3或许会友好很多。
```

# 心路历程 
**具体记录一下我进行研究项目的想法**  
首先的话是觉得这个模型很厉害，能够做到一个比较好的中心化去边界框的任务，想了解一下具体怎么实现的。由于自己深度学习方面涉猎甚少，而且很久没接触了，就想着拿这个项目练手。  
秉持着一种做事自顶向下的态度，决定首先先把这个项目跑起来吧，就看着Readme开始了环境的搭建，安装依赖，对着这个下载预训练模型，跑demo.py，但是出了点问题。找不到module，查找发现是没给添加环境变量，云服务器添加不了，没办法从自己电脑添加试试。跑了之后发现，啥区别没有，照样该报错报错。之后查了查github里面的issue发现有人问题一样，解决方法上的话，我认为主要就是没有按照他的教程去做，如果正常的话，进行INSTALL操作后完全可以实现的。顺着INSTALL.MD进行了操作。 
综合考虑是自己对于setup文件不够了解，需要花时间去学习setup相关的知识，让自己在遇到相关的问题的时候也能有足够的知识储备去解决问题。  
附上一个学习 setuptools 的网址：[quickstart](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)