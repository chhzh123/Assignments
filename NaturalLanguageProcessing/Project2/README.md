# 自然语言处理项目 - 机器翻译

任务：构建带Attention的Seq2Seq模型，实现中文到英文的翻译


## 文件组织

* 数据集为`dataset_10000`与`dataset_100000`，里面含有训练集、验证集和测试集，均为对应的平行语料（每一行为一句中文或一句英文）
* 预训练的模型在`checkpoint`文件夹中
* 运行日志可见`*.log`文件
* **实验报告请见`main.pdf`**


## 运行方式

1. 安装依赖关系包

```bash
$ pip install -r requirement.txt
```

注意电脑需提前安装好GPU和相应的CUDA环境，本项目采用Pytorch作为编程框架。

2. 模型训练、预测与评估

推荐打开`seq2seq.ipynb`采用Jupyter Notebook交互界面进行操作。或使用下述指令一键运行

```bash
$ python seq2seq.py
```

## 参考资料
1. Stanford [CS224n](http://web.stanford.edu/class/cs224n/): Natural Language Processing with Deep Learning
2. Pytorch, [seq2seq translation tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
3. Practical Pytorch, [Batched seq2seq](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb)
4. Sutskever Ilya, Vinyals Oriol, and Le Quoc V, *Sequence to Sequence Learning with Neural Networks*, NeurIPS, 2014
5. Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR, 2015
6. Minh-Thang Luong, Hieu Pham, and Christopher D. Manning, *Effective Approaches to Attention-based Neural Machine Translation*, EMNLP, 2015
7. Papineni Kishore, Roukos Salim, Ward Todd, and Zhu Wei-Jing, *BLEU: a method for automatic evaluation of machine translation*, ACL, 2002