from transformers import BertModel
import torch as t
from config.KBConfig import *


class BasicBertModel(t.nn.Module):
    def __init__(self, pretrain_bert_path):
        super(BasicBertModel, self).__init__()
        bert_config = BertConfig.from_pretrained(pretrain_bert_path) # 获取预训练模型的参数
        self.bert_model = BertModel.from_pretrained(pretrain_bert_path, config=bert_config)
        self.out_linear_layer = t.nn.Linear(bert_config.hidden_size, bert_output_dim) # 参数的值分别为(768,300)，表示创建一个全连接层，第一个参数表示输入的形状，第二个参数表示输出的形状
        self.dropout = t.nn.Dropout(p=0.1) # 设置丢弃率，不激活某些神经元，防止过拟合

    def forward(self, tids, masks):
        bert_out = self.bert_model(input_ids=tids, attention_mask=masks)
        last_hidden_state = bert_out.last_hidden_state
        cls = last_hidden_state[:, 0]
        output = self.dropout(cls)
        output = self.out_linear_layer(output)
        # output = t.nn.functional.normalize(output, p=2, dim=-1)
        return output
