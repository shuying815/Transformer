import torch
import os
from transformers import GPT2Tokenizer

# -----------------------------------------------------------------------------
# WikiText-2数据集
# path: 数据集根路径
# -----------------------------------------------------------------------------
class WikiText:
    def __init__(self, path):
        # 读取数据，.tokens文件中是已分词的文本
        path = os.path.join(path, 'wikitext-2')
        with open(os.path.join(path, 'wiki.train.tokens'), 'r') as f:
            train_data = f.read()
        with open(os.path.join(path, 'wiki.test.tokens'), 'r') as f:
            test_data = f.read()
        with open(os.path.join(path, 'wiki.valid.tokens'), 'r') as f:
            val_data = f.read()

        # 构建tokenizer，使用GPT2词汇表
        self.tokenizer = GPT2Tokenizer.from_pretrained('./gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 编码：numericalize
        self.train_txt = self.encoding(train_data)
        self.test_txt = self.encoding(test_data)
        self.val_txt = self.encoding(val_data)

    # data: 已分词的token数据
    def encoding(self, data):
        return [self.tokenizer.convert_tokens_to_ids(word) for word in data]

    # 将文本数据转换成指定形式
    # data：数据
    # batch_size: 批量大小
    def batchify(self, data, batch_size):
        n_batch = len(data) // batch_size
        numericalized_data = data[:n_batch * batch_size]
        data = torch.tensor(numericalized_data).view(batch_size, -1).t().contiguous() # [seq_len, batch_size]
        return data

    # 获取一个batch的数据
    # source: 源数据
    # i: 从第i个token开始读取
    def get_batch(self, source, i):
        bptt = 35  # 一个句子种最多包含35个单词
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len]
        return data, target

    def get_data(self, batch_size):
        train_data = self.batchify(self.train_txt, batch_size)
        val_data = self.batchify(self.val_txt, batch_size)
        test_data = self.batchify(self.test_txt, batch_size)
        return train_data, val_data, test_data

