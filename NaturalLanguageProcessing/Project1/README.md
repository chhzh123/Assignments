# 自然语言处理项目 - 文本预测

任务：补全句子中挖空的词语（以`[MASK]`标识），如下面的例子
```
因为刷脸支付也得去打开手机接收验证码，所以还不如直接扫[MASK]更直接更方便。（标准答案：二维码）
```

主要分为数据获取、预处理和模型生成与预测三个部分。

* 所有新闻文本原始数据从[网易科技新闻](https://tech.163.com/)上获取
* 采用n-gram（统计模型）和LSTM（神经网络模型）两个模型进行预测


## 文件组织

* 测试集为`questions.txt`与`answer.txt`，共100条问句
* 预训练的模型在`checkpoint`文件夹中（包括n-gram的词表及LSTM模型参数）
* 预测的结果见`myanswer`文件夹，每一行为一个答案，同一行中有多个预测值
* 运行日志可见`*.log`文件
* **实验报告请见`main.pdf`**


## 运行方式

程序都已附在当前文件夹，欲执行程序，可直接通过`make`一键完成数据获取、处理、训练、预测全过程。

若要分步执行，请按照以下步骤进行。

1. 安装依赖关系包

```bash
$ pip install -r requirement.txt
```

2. 数据采集与预处理

数据采集的程序为`spider.py`，预处理程序为`cut.py`，可以直接通过下列指令执行

```bash
$ make data
```

3. 模型生成与预测

N-gram模型可见`ngram.py`文件，使用下述指令运行，其中N和K为命令行参数

```bash
$ python ngram.py --N 3 --K 5
```

LSTM模型可见`lstm.ipynb`文件或由jupyter notebook生成的`lstm.py`文件，可选超参数见程序定义

```bash
$ python lstm.py
```

## 参考资料
1. [中文常用停止词列表](https://github.com/goto456/stopwords)
2. [Text Generation With Pytorch](https://machinetalk.org/2019/02/08/text-generation-with-pytorch/)