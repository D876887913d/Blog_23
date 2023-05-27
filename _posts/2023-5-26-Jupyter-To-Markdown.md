---
layout: posts
title:  "Jupyter To markdown"
date:   2023-05-26 7:46:22 +0800
categories: jekyll update
---

# 相关库依赖安装
```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
pip install jupyter_nbextensions_configurator
jupyter nbextensions_configurator enable --user
```

# 实际进行转换
`jupyter nbconvert file_learn.ipynb --to markdown`