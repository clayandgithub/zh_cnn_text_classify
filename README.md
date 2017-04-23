# 基于cnn的中文文本分类算法

## 简介
参考[IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW]((http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/))实现的一个简单的卷积神经网络，用于中文文本分类任务（此项目使用的数据集是中文垃圾邮件识别任务的数据集）。

## 区别
原博客实现的cnn用于英文文本分类，没有使用word2vec来获取单词的向量表达，而是在网络中添加了embedding层来来获取向量。
而此项目则是利用word2vec先获取中文测试数据集中各个<strong>字</strong>的向量表达，再输入卷积网络进行分类。

## 运行方法

### 训练
run `python train.py` to train the cnn with the <strong>spam and ham files (only support chinese!)</strong> (change the config filepath in FLAGS to your own)

### 在tensorboard上查看summaries
run `tensorboard --logdir /{PATH_TO_CODE}/runs/{TIME_DIR}/summaries/` to view summaries in web view

### 测试、分类
run `python eval.py --checkpoint_dir /{PATH_TO_CODE/runs/{TIME_DIR}/checkpoints}`
如果需要测试准确率，需要指定对应的标签文件。
如果需要分类自己提供的文件，请更改相关输入参数
