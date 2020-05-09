## Chinese NER using Bert
基于pytorch版本的bert的中文NER

### 0.准备
#### 0.1 依赖
1. PyTorch=1.1.0+
2. cuda=9.0
3. python3.6+

#### 0.2 pytorch 版本的预训练模型下载
pytorch版的预训练模型可以在在这里下载：

config: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json

vocab: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt

model: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin

也可以根据这个代码将tf版本转换为pytorch版本
https://github.com/JackKuo666/convert_tf_bert_model_to_pytorch


预训练模型要这样存放：
```text
├── prev_trained_model
|  └── bert_base
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── vocab.txt
|  |  └── ......
```

#### 0.3 自定义数据集或者CLUENER数据集下载（可选）

```text
├── BERT-NER-Pytorch
|  └── datasets
|  |  └── cluener
|  |  └── cner
|  |  └── ......
```

1. cner: datasets/cner  [已经有了]
2. CLUENER: https://github.com/CLUEbenchmark/CLUENER
3. 你也可以自己制作数据集放在这里

### 1. 模型列表
这里有三个模型：

1. BERT+Softmax
2. BERT+CRF
3. BERT+Span

### 2.数据输入格式

1.输入格式是：`BIOS`标注策略，但是如果有`BME`格式的会将`ME`转换为`I`.

2.中文数据集是每个字符存放一行，标签与字符之间有一个空格，句子之间有一个空行。

```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
```

### 3.运行代码

#### 3.1.默认`cner`是可以训练的
```
sh scripts/run_ner_crf.sh
```
默认预测:
```
sh scripts/run_ner_crf_predict.sh
```
#### 3.2.如果要运行其他数据集需要修改：

##### 3.2.1.修改```run_ner_crf.sh```中的`TASK_NAME="cner"`；
##### 3.2.2.仿照`/processors/ner_seq.py`中的`class CnerProcessor(DataProcessor):`构造自己的Processor（只需要修改其中的`    def get_labels(self):`为新数据集的labels），同时注册该类：
```
ner_processors = {
    "cner": CnerProcessor,
    "datasets下文件夹名字": 新增类名,
    'cluener':CluenerProcessor
}
```
### 其他
另外需要注意的是，这个项目没有提供默认的设置GPU的接口，可以在`run_ner_crf.py`中设置：
```
def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    args = get_argparse().parse_args()
```

---
# 以下为原作者的结果：

### CLUENER result

The overall performance of BERT on **dev**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.7897     | 0.8031     | 0.7963    |
| BERT+CRF     | 0.7977 | 0.8177 | 0.8076 |
| BERT+Span    | 0.8132 | 0.8092 | 0.8112 |
| BERT+Span+adv    | 0.8267 | 0.8073 | **0.8169** |
| BERT-small(6 layers)+Span+kd    | 0.8241 | 0.7839 | 0.8051 |
| BERT+Span+focal_loss    | 0.8121 | 0.8008 | 0.8064 |
| BERT+Span+label_smoothing   | 0.8235 | 0.7946 | 0.8088 |

### ALBERT for CLUENER

The overall performance of ALBERT on **dev**:

| model  | version       | Accuracy(entity) | Recall(entity) | F1(entity) | Train time/epoch |
| ------ | ------------- | ---------------- | -------------- | ---------- | ---------------- |
| albert | base_google   | 0.8014           | 0.6908         | 0.7420     | 0.75x            |
| albert | large_google  | 0.8024           | 0.7520         | 0.7763     | 2.1x             |
| albert | xlarge_google | 0.8286           | 0.7773         | 0.8021     | 6.7x             |
| bert   | google        | 0.8118           | 0.8031         | **0.8074**     | -----            |
| albert | base_bright   | 0.8068           | 0.7529         | 0.7789     | 0.75x            |
| albert | large_bright  | 0.8152           | 0.7480         | 0.7802     | 2.2x             |
| albert | xlarge_bright | 0.8222           | 0.7692         | 0.7948     | 7.3x             |

### Cner result

The overall performance of BERT on **dev(test)**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.9586(0.9566)     | 0.9644(0.9613)     | 0.9615(0.9590)     |
| BERT+CRF     | 0.9562(0.9539)     | 0.9671(**0.9644**) | 0.9616(0.9591)     |
| BERT+Span    | 0.9604(**0.9620**) | 0.9617(0.9632)     | 0.9611(**0.9626**) |
| BERT+Span+focal_loss    | 0.9516(0.9569) | 0.9644(0.9681)     | 0.9580(0.9625) |
| BERT+Span+label_smoothing   | 0.9566(0.9568) | 0.9624(0.9656)     | 0.9595(0.9612) |
