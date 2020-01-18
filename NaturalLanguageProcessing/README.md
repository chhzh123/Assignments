# Natural Language Processing (NLP)

This is the repository of chhzh123's assignments of *Database Systems* - Fall 2019 @ SYSU lectured by *Xiaojun Quan*.

There are mainly two projects in this course and details can be found in each folder. (The first homework is to be familiar with the Chinese word-cutting tools.)

## Project 1 - Text Prediction
Task: Fill in the blank word in the sentence (labeled with `[MASK]`), for example,
```
因为刷脸支付也得去打开手机接收验证码，所以还不如直接扫[MASK]更直接更方便。（标准答案：二维码）
```

This project needs to be divided into three parts.
1. Data acquisition
All the news original data are grabbed from [NetEast Tech](https://tech.163.com/).

2. Data preprocessing

3. Model generation and prediction
**n-gram** and **LSTM** models are used to predict the text.

## Project 2 - Machine Translation
Task: Build the **seq2seq model with attention** mechanism and translate Chinese to English.