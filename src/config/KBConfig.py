import argparse
import datetime
import os
import random
import sys
import torch as t
import numpy as np
from transformers import BertConfig

from tools import Logger
# 初始化参数和保存的文件路径
seed = 11037
random.seed(seed)
t.manual_seed(seed)
t.cuda.manual_seed_all(seed)
np.random.seed(seed)

time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')[:-3] # 当前时间
version_str = time_str[:10]
run_file_name = sys.argv[0].split('/')[-1].split('.')[0] # 当前运行的文件名

parser = argparse.ArgumentParser() # argparse用于解析命令行参数
parser.add_argument('--mode', type=str, default='OpenEA') # 定义参数选项
parser.add_argument('--log', action='store_true')
# ================= Dataset ===============================================
data_group = parser.add_argument_group(title='General Dataset Options') # 创建参数组，将相关的命令行参数组织在一起
data_group.add_argument('--result_root', type=str, default='../outputs')
data_group.add_argument('--functionality', action='store_true')
data_group.add_argument('--blocking', action='store_true')
data_group.add_argument('--pretrain_bert_path', type=str)
data_group.add_argument('--basic_bert_path', type=str)
data_group.add_argument('--datasets_root', type=str)
data_group.add_argument('--relation', action='store_true')
# =========================================================================
# ================= OpenEA ================================================
openea_group = parser.add_argument_group(title='OpenEA Dataset Options')
openea_group.add_argument('--dataset', type=str, metavar='dataset path')
openea_group.add_argument('--fold', type=int, default=0)
# =========================================================================
train_group = parser.add_argument_group(title='Train Options')
train_group.add_argument('--gpus', type=str)
train_group.add_argument('--version', type=str)

args = parser.parse_args() # 解析命令行参数，并返回解析结果的对象
seq_max_len = 128 # 序列最大长度
bert_output_dim = 300 # bert模型输出维度
PARALLEL = False # 是否使用并行计算
DEBUG = False # 是否启用调试模式
SCORE_DISTANCE_LEVEL, MARGIN = 2, 1


if args.version is not None:
    version_str = args.version
if args.relation:
    version_str += '-relation'

# ================= OpenEA ================================================
dataset_name = args.dataset
dataset_home = os.path.join(args.datasets_root, dataset_name)

ds = dataset_name.split('_') # ds:['zh','en']
log_name = '-'.join((time_str, dataset_name, str(args.fold), run_file_name, str(seq_max_len))) # 日志名称
result_home = '/'.join((args.result_root, version_str, dataset_name)) # 运行结果存放路径
if not os.path.exists(result_home): # 判断路径是否存在
    os.makedirs(result_home)
need_log = args.log
log_path = os.path.join(result_home, 'logs') # 日志路径
if need_log:
    Logger.make_print_to_file(name=log_name + '-add.txt', path=log_path) # 将日志输出到文件中


class Dataset:
    def __init__(self, no):
        self.name = ds[no-1]
        self.attr = self.triples('attr', no) # zh/en的属性三元组路径 -- '../data/zh_en/attr_triples_1'
        self.rel = self.triples('rel', no)# zh/en的关系三元组路径 -- '../data/zh_en/rel_triples_1'
        self.entities_out = self.outputs_tab('entities', no) # zh/en实体存放路径
        self.literals_out = self.outputs_tab('literals', no) # zh/en属性值
        self.properties_out = self.outputs_tab('properties', no) # zh/en属性名称
        self.relations_out = self.outputs_python('relations', no)# zh/en关系存放路径
        self.table_out = self.outputs_csv('properties', no) # 表格形式存放实体，属性和属性值
        self.seq_out = self.outputs_tab('sequence_form', no) # 序列形式存放实体，属性值
        self.tokens_out = self.outputs_python('tokens', no) # token形式的实体，属性值
        self.tids_out = self.outputs_python('tids', no) # id形式的属性值
        self.token_freqs_out = self.outputs_python('token_freqs', no)
        self.facts_out = self.outputs_python('facts', no)
        self.case_study_out = self.outputs_python('case_study', no)
    # 返回对象的字符串表示形式
    def __str__(self):
        return 'Dataset{name: %s, rel: %s, attr: %s}' % (self.name, self.rel, self.attr)

    @staticmethod
    def triples(name, no):
        file_name = '_'.join((name, 'triples', str(no)))
        return os.path.join(dataset_home, file_name)

    @staticmethod
    def outputs_tab(name, no):
        file_name = '_'.join((name, 'tab', str(no))) + '.txt'
        return os.path.join(result_home, file_name)

    @staticmethod
    def outputs_csv(name, no):
        file_name = '_'.join((name, 'csv', str(no))) + '.csv'
        return os.path.join(result_home, file_name)

    @staticmethod
    def outputs_python(name, no):
        file_name = '_'.join((name, 'python', str(no))) + '.txt'
        return os.path.join(result_home, file_name)


dataset1 = Dataset(1) # 1代表zh，2代表en
dataset2 = Dataset(2)


class OEAlinks:
    def __init__(self, fold):
        self.block = self.result_path('block', 0)
        # if fold > 0:
        self.train = self.links_path('train', fold)
        self.valid = self.links_path('valid', fold)
        self.test = self.links_path('test', fold)
        self.truth = '/'.join((dataset_home, 'ent_links')) # 种子实体对路径
        self.model_save = '/'.join((result_home, log_name, 'basic_bert_model.pkl')) # bert模型路径
        self.rel_model_save = '/'.join((result_home, log_name, 'rel_model.pkl')) # 关系模型存储路径
        self.case_study_out_1 = '/'.join((result_home, log_name, 'case_study_1.txt')) # zh关系值
        self.case_study_out_2 = '/'.join((result_home, log_name, 'case_study_2.txt')) # en关系值
        if args.basic_bert_path is None:
            self.kb_prop_emb_1 = '/'.join((result_home, log_name, '_'.join((str(fold), 'kb_prop_emb_1.pt'))))
            self.kb_prop_emb_2 = '/'.join((result_home, log_name, '_'.join((str(fold), 'kb_prop_emb_2.pt'))))
        else:
            self.kb_prop_emb_1 = '/'.join((os.path.dirname(args.basic_bert_path), '_'.join((str(fold), 'kb_prop_emb_1.pt'))))
            self.kb_prop_emb_2 = '/'.join((os.path.dirname(args.basic_bert_path), '_'.join((str(fold), 'kb_prop_emb_2.pt'))))
            self.rel_model_load = '/'.join((os.path.dirname(args.basic_bert_path), 'rel_model.pkl'))
        pass

    @staticmethod
    def links_path(train_type, fold_num):
        return os.path.join(dataset_home, '/'.join(('721_5fold', str(fold_num), '_'.join((train_type, 'links')))))

    @staticmethod
    def result_path(name, fold_num):
        return os.path.join(result_home, '_'.join((name, str(fold_num), '.txt')))


links = OEAlinks(args.fold)
#？
if args.mode == 'KB':
    functionality_control = True
else:
    functionality_control = args.functionality
functionality_threshold = 0.9

print('time str:', time_str)
print('run file:', run_file_name)
print('args:')
print(args)
print('log path:', os.path.abspath(log_path))
print('log:', need_log)

print('dataset1:', dataset1)
print('dataset2:', dataset2)
print('result_path:', result_home)
