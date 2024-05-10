from collections import Counter
from itertools import chain

from transformers import BertTokenizer

from config.KBConfig import *
from tools import FileTools
from tools.Announce import Announce
from tools.MultiprocessingTool import MPTool


class BertDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        datas = self.load_data(BertDataLoader.line_to_feature)
        self.save_data(datas)
        self.save_token_freq(datas)

    def load_data(self, line_solver):
        data_path = self.dataset.seq_out# 只将属性值进行了分词，没有对关系分词
        print(Announce.doing(), 'load BertTokenizer')
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path) # 加载了预训练的 BERT 模型的 tokenizer，使用它来对文本进行分词等处理
        print(Announce.done())
        with open(data_path, 'r', encoding='utf-8') as rfile:
            datas = MPTool.packed_solver(line_solver, tokenizer=tokenizer).send_packs(rfile).receive_results()  # 对保存的seq数据进行分词处理，data包含三部分，id，分词向量，
            for eid, tokens, tids in datas:
                print((eid,tokens,tids))
        return datas  # datas是经过tokenizer编码后的向量形式，datas = [(0, ['il', '-', '2', 'mr', '##na', 'rna'], [10145, 118, 123, 12854, 10206, 31655]),...]

    @staticmethod
    def line_to_feature(line: str, tokenizer: BertTokenizer):
        eid, text = line.strip('\n').split('\t')
        tokens = tokenizer.tokenize(text) #对text文本进行分词操作
        tid_seq = tokenizer.convert_tokens_to_ids(tokens) # 将token转换为对应的id，就是当前token对应的位置
        return int(eid), tokens, tid_seq

    def save_data(self, datas):
        tokens_path = self.dataset.tokens_out
        tids_path = self.dataset.tids_out
        tids = [(eid, tids) for eid, tokens, tids in datas] # tids存储的是 -- id：token值编码（seq分解后的数据）
        tokens = [(eid, tokens) for eid, tokens, tids in datas]
        FileTools.save_list_p(tids, tids_path)  # 保存每条seq分词后的token对应的id tids_python_.txt
        FileTools.save_list_p(tokens, tokens_path) # 保存每条seq分词后的token tokens_python_.txt
        # return tokens, tids

    @staticmethod
    def load_saved_data(dataset):
        tids_path = dataset.tids_out
        tids = FileTools.load_list_p(tids_path)
        return tids

    def save_token_freq(self, datas) -> dict:
        freq_path = self.dataset.token_freqs_out
        tokens = [tokens for eid, tokens, tids in datas]
        tids = [tids for eid, tokens, tids in datas]
        tokens = list(chain.from_iterable(tokens)) # 将一个列表中嵌套的列表（或其他可迭代对象）展开，返回一个扁平化的迭代器。
        tids = list(chain.from_iterable(tids))
        results = [(token, tid) for token, tid in zip(tokens, tids)] # 列表形式，每个元组由token及其对应的id组成
        r_counter = Counter(results) # 用于计算results中每个元素出现的频率，results中每个元素是元组
        # FileTools.save_dict(r_counter, freq_path)
        r_dict = dict(r_counter)
        r_list = sorted(r_dict.items(), key=lambda x: x[1], reverse=True)
        FileTools.save_list_p(r_list, freq_path) # 保存每个分词token的id及总共出现的次数 token_freqs_python_.txt
        return r_dict

    @staticmethod
    def load_freq(dataset):
        freq_path = dataset.token_freqs_out
        print(Announce.printMessage(), 'load:', freq_path)
        freq_list = FileTools.load_list_p(freq_path)
        freq_dict = {key: value for key, value in freq_list}
        return freq_dict
