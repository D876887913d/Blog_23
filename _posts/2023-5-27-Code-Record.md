---
layout: posts
title:  "对数据进行 Transform 具体构建的方法"
date:   2023-05-27 15:17:31 +0800
categories: jekyll update
---


# 官方教程
[TorchVision.Transforms](https://pytorch.org/vision/stable/transforms.html)


# Transform 类构建

如下是来源于JetSeg进行Transform的一个类：
```python
from torchvision import transforms as T
class Transforms:
    def __init__(self, img=None, mask=None, img_size=None):
        self.img = img
        self.mask = mask
        self.img_size = img_size

        if img is None and mask is None:
            raise ValueError("[ERROR] Both the image and the mask cannot be None")

        # Build transforms
        self.transform_img, self.transform_mask = self.build_transform()

    # 同时处理多个不同的数据集的时候，采用的图像的RGB均值方差不同。
    def get_normalize_info(self, dataset_name="camvid"):
        if dataset_name.lower() == "camvid":
            mean = [0.4132, 0.4229, 0.4301]
            std = [0.1096, 0.1011, 0.0963]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        return mean, std

    # 判断图像是不是Image.Image类型的
    def is_pil_image(self, img):
        return isinstance(img, Image.Image)

    # 构造Transforms序列，即构造要进行的所有的Transforms。
    def build_transform(self):
        ops_img = []
        ops_mask = []

        if self.img is not None:

            # Resize image
            if self.img_size is not None:
                ops_img.append(T.Resize(self.img_size))

            # Convert to tensor and normalize image
            ops_img.append(T.ToTensor())

            # Get Dataset Normalization
            mean, std = self.get_normalize_info()
            ops_img.append(T.Normalize(mean, std))

        if self.mask is not None:

            # Resize mask
            if self.img_size is not None:
                # 采用NEAREST插值法进行图像缩放
                ops_mask.append(T.Resize(self.img_size,
                                         interpolation=Image.NEAREST))

            # Convert to tensor
            ops_mask.append(T.ToTensor())

            # Convert to PIL image
            ops_mask.append(T.ToPILImage())

        return T.Compose(ops_img), T.Compose(ops_mask)

    def apply_transform(self):
        # 分别对img,mask进行transforms操作.
        if self.img is not None:
            img = self.transform_img(self.img)
        else:
            img = None

        if self.mask is not None:
            mask = self.transform_mask(self.mask)
        else:
            mask = None

        return img, mask
```

# 调用方法:
`
Transforms(img=img, mask=mask, img_size=self.img_size)
`  
将图像输入函数,mask输入函数,将图像尺寸输入,即可进行构造函数的构建.

```python
# 获取一个图片以及其掩码
img, mask = self.get_sample(idx)
# 构造Tranformer类
transformer = Transforms(img=img, mask=mask, img_size=self.img_size)
# 调用Transforms获取处理后的函数。
img, rgb_mask = transformer.apply_transform()
```